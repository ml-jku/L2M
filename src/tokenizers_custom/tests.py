from src.tokenizers_custom import make_tokenizer


if __name__ == "__main__":
    import torch

    tok_kwargs = {"vocab_size": 256}
    mulaw = make_tokenizer('mulaw', tok_kwargs)
    minmax = make_tokenizer('minmax', tok_kwargs)
    # minmax2 = make_tokenizer('minmax2', tok_kwargs)
    
    tok_kwargs = {"vocab_size": 256, "shift": 18}
    mulaw_shift = make_tokenizer('mulaw', tok_kwargs)
    minmax_shift = make_tokenizer('minmax', tok_kwargs)
    
    x = torch.linspace(-1, 1, 20)

    print("------ MuLaw ------")
    tokens = mulaw.tokenize(x).clone()
    x_inv = mulaw.inv_tokenize(tokens).clone()
    tokens_inv = mulaw.tokenize(x_inv).clone()
    print(x)
    print(x_inv)
    print(tokens)
    print(tokens_inv)
    assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"

    print("\n\n------ MinMax ------")
    tokens = minmax.tokenize(x).clone()
    x_inv = minmax.inv_tokenize(tokens).clone()
    tokens_inv = minmax.tokenize(x_inv).clone()
    print(x)
    print(x_inv)
    print(tokens)
    print(tokens_inv)
    assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"
    
    print("------ MuLaw shift ------")
    tokens = mulaw_shift.tokenize(x).clone()
    x_inv = mulaw_shift.inv_tokenize(tokens).clone()
    tokens_inv = mulaw_shift.tokenize(x_inv).clone()
    print(x)
    print(x_inv)
    print(tokens)
    print(tokens_inv)
    assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"
    
    print("\n\n------ MinMax shift ------")
    tokens = minmax_shift.tokenize(x).clone()
    x_inv = minmax_shift.inv_tokenize(tokens).clone()
    tokens_inv = minmax_shift.tokenize(x_inv).clone()
    print(x)
    print(x_inv)
    print(tokens)
    print(tokens_inv)
    assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"

    # print("\n\n------ MinMax 2------")
    # tokens = minmax2.tokenize(x)
    # x_inv = minmax2.inv_tokenize(tokens)
    # tokens_inv = minmax2.tokenize(x_inv)
    # print(x)
    # print(x_inv)
    # print(tokens)
    # print(tokens_inv)
    # assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"

    for i in [10, 20, 50, 100, 1000, 10000]:
        x = torch.linspace(-1, 1, i)

        # mulaw
        tokens = mulaw.tokenize(x).clone()
        x_inv = mulaw.inv_tokenize(tokens).clone()
        tokens_inv = mulaw.tokenize(x_inv).clone()
        assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"

        # minmax
        tokens = minmax.tokenize(x).clone()
        x_inv = minmax.inv_tokenize(tokens).clone()
        tokens_inv = minmax.tokenize(x_inv).clone()
        assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"

        # # minmax2
        # tokens = minmax2.tokenize(x)
        # x_inv = minmax2.inv_tokenize(tokens)
        # tokens_inv = minmax2.tokenize(x_inv)
        # assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"

        # mulaw shift
        tokens = mulaw_shift.tokenize(x).clone()
        x_inv = mulaw_shift.inv_tokenize(tokens).clone()
        tokens_inv = mulaw_shift.tokenize(x_inv).clone()
        assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"

        # minmax shift
        tokens = minmax_shift.tokenize(x).clone()
        x_inv = minmax_shift.inv_tokenize(tokens).clone()
        tokens_inv = minmax_shift.tokenize(x_inv).clone()
        assert torch.allclose(tokens, tokens_inv), "Tokens are not the same"
