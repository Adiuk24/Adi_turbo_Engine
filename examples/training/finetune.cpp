#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n",
                __func__);
        params.use_mmap = false;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    // load the model and apply lora adapter, if any
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    struct llama_adapter_lora * lora_adapter = nullptr;
    if (params.qat.lora_train) {
        LOG_INF("%s: Initializing new LoRA adapter with rank %d\n", __func__, params.qat.lora_rank);
        lora_adapter = llama_adapter_lora_init_new(model, params.qat.lora_rank);
        if (!lora_adapter) {
            LOG_ERR("%s: Failed to initialize LoRA adapter\n", __func__);
            return 1;
        }
        float scale = 1.0f;
        llama_set_adapters_lora(ctx, &lora_adapter, 1, &scale);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    std::string train_data = params.prompt;
    if (!params.qat.data_path.empty()) {
        auto read_file = [](const std::string& path) {
            FILE * fp = fopen(path.c_str(), "rb");
            if (!fp) return std::string();
            fseek(fp, 0, SEEK_END);
            size_t size = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            std::string content(size, ' ');
            fread(&content[0], 1, size, fp);
            fclose(fp);
            return content;
        };
        train_data = read_file(params.qat.data_path);
        if (train_data.empty()) {
            LOG_ERR("%s: failed to read %s\n", __func__, params.qat.data_path.c_str());
            return 1;
        }
    }

    std::vector<llama_token> tokens  = common_tokenize(ctx, train_data, true);
    
    // Ensure we have enough tokens to avoid bad_alloc from negative ndata
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    if ((int64_t)tokens.size() <= ne_datapoint + 1) {
        LOG_ERR("%s: training data is too small! Context size is %d, but data has only %zu tokens.\n", __func__, (int)ne_datapoint, tokens.size());
        return 1;
    }
    ggml_opt_dataset_t       dataset = common_opt_dataset_init(ctx, tokens, ne_datapoint / 2);

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    common_init_result_ptr teacher_init = nullptr;
    if (params.qat.distill && !params.teacher.path.empty()) {
        common_params tparams = params;
        tparams.model = params.teacher;
        tparams.qat.lora_train = false;
        teacher_init = common_init_from_params(tparams);
        if (!teacher_init || !teacher_init->model()) {
            LOG_ERR("%s: unable to load teacher model\n", __func__);
            return 1;
        }
        LOG_INF("%s: successfully loaded teacher model from %s\n", __func__, params.teacher.path.c_str());
    }

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/params.qat.lora_train ? llama_opt_param_filter_lora : llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
        /*ctx_teacher     =*/teacher_init ? teacher_init->context() : nullptr,
        /*distill_temp    =*/params.qat.distill_temp,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    if (lora_adapter) {
        std::string lora_out = params.qat.lora_out.empty() ? "lora-distilled.gguf" : params.qat.lora_out;
        LOG_INF("%s: saving Distilled LoRA adapter to %s\n", __func__, lora_out.c_str());
        if (!llama_adapter_lora_save(lora_adapter, lora_out.c_str())) {
            LOG_ERR("%s: failed to save LoRA adapter\n", __func__);
        }
    } else {
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    llama_backend_free();

    return 0;
}
