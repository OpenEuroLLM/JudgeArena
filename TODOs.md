TODOs:
* push on pypi
* document on the fly evaluations with custom prompt
* support MT-bench
* handle errors
* CI [high/large]
* implement CI judge option
* implement domain filter in CI (maybe pass a regexp by column?)
* report cost?

Done:
* support alpaca-eval
* support arena-hard
* test together judge
* local env variable to set paths
* tqdm callback with batch
* support loading local completions
* support dumping outputs [medium/small]
* test LlamaCpp [medium/small]
* test openai judge [medium/small]
* test vLLM judge [medium/small]
* CLI launcher [medium/large]
* put contexts in HF dataset [high/small]
* mAH: instruction loader [DONE]
* mAH: generate instructions for two models [DONE]
* mAH: make comparison [DONE]
* mAH: support using all languages at once [high/medium]
* unit-test
* add details to example to generate and evaluate completions
* installation instructions
* generate proper json of results
* improve login of results
* rename {lang}-contexts to fluency-{lang}
  * needs renaming of datasets
  * update in code and doc
* support instruction evaluation
  * small refactor `annotate` needs to return just the judge completion, not the parsed one
  * perhaps change to `annotate_pair` and `annotate_single`
  * then provide example
* support evaluation with input swap
