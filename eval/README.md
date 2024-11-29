# Evaluation of Insurance Chat

Format of a test case:

```
{
    setup: {
        current_document: string,
    },
    messages?: {
        content: string,
        assert_response_includes?: string[], # Check that a response contains all strings in list,
        open_ended_eval: string, # An open-ended condition for model-graded eval
    }[],
    test?: string, # An open-ended condition for full conversational eval, ignored if messages are provided
}
```

See `./tests/basic.json` for an example.

## Instructions

Run a test with

```
# To run all tests
$ uv run eval --all

# To run only tests in plans.json
$ uv run eval --test plans.json  

# To run only a subset of tests in plans.json
$ uv run eval --test plans.json --labels aia ge prudential raffles

# To view thoughts and conversation use the --verbose flag 
$ uv run eval --test plans.json --verbose
```
