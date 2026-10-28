// Optional fixture generator, not part of MDMA's build or runtime.
// Compile against the pinned Prism reference described in doc/bonsai.md.
#include "llama.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

static void ids(const std::vector<llama_token> & values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("MODEL PROMPT MAX_TOKENS required");
        const int limit = std::stoi(argv[3]);
        if (limit < 1 || limit > 64) throw std::runtime_error("MAX_TOKENS must be 1..64");
        Backend backend;
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)>
            model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::string prompt = argv[2];
        const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        if (count <= 0 || count + limit > 256) throw std::runtime_error("invalid prompt/context");
        std::vector<llama_token> input(count);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), count, true, true) != count)
            throw std::runtime_error("tokenization failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = 256;
        cp.n_ubatch = 1;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)>
            ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx || llama_decode(ctx.get(), llama_batch_get_one(input.data(), input.size())) != 0)
            throw std::runtime_error("prefill failed");
        std::vector<llama_token> output;
        for (int i = 0; i < limit; ++i) {
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            llama_token token = 0;
            for (llama_token j = 1; j < llama_vocab_n_tokens(vocab); ++j)
                if (logits[j] > logits[token]) token = j;
            if (llama_vocab_is_eog(vocab, token)) break;
            output.push_back(token);
            if (i + 1 < limit && llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) != 0)
                throw std::runtime_error("decode failed");
        }
        std::cout << "{\"prompt_tokens\":";
        ids(input);
        std::cout << ",\"tokens\":";
        ids(output);
        std::cout << "}\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
