## cs598-automated-icd-coding

Two directories: final_codes (latest version of code files) and draft_codes (old version of files i.e before draft submission)

### Preprocessing
Load the pickle raw data of MIMIC-III to execute preprocessing steps and dump the resultant dataframe as a pickle file.
<pre><code>python final_codes/preprocess.py <path_to_raw_data_pickle></code></pre>


## Train/Evaluate
Load the preprocessed data from above step to start training for one of the given nine models:

| Model      | Command |
| ----------- | ----------- |
| BoT Baseline      | <pre><code> python final_codes/<model.py/parallel_model.py> --model bot </code></pre> |
| BoT Baseline w/ Cat 0 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model bot --category</code></pre> |
| BoT Baseline w/ Cat 1 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model bot --category --fullyConnected 120</code></pre> |
| BoT Baseline w/ Cat 2 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model bot --category --fullyConnected 200 100</code></pre> |
| BoT Baseline w/ Cat 3 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model bot --category --fullyConnected 230 170 110</code></pre> |
| CNN Baseline      | <pre><code> python final_codes/<model.py/parallel_model.py> --model cnn </code></pre> |
| CNN Baseline w/ Cat 0 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model cnn --category </code></pre> |
| CNN Baseline w/ Cat 1 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model cnn --category --fullyConnected 3000 </code></pre> |
| CNN Baseline w/ Cat 2 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model cnn --category --fullyConnected 3000 500  </code></pre> |
| CNN Baseline w/ Cat 3 Dense   | <pre><code> python final_codes/<model.py/parallel_model.py> --model cnn --category --fullyConnected 3000 700 200 </code></pre> |
| CNN 3-Conv1D   | <pre><code> python final_codes/<model.py/parallel_model.py> --model cnn3 </code></pre> |
