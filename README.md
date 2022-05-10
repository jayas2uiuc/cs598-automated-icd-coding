## CS598-Automated-Icd-Coding

This is the code repository for the paper <code>Vitor Pereira, Sérgio Matos, and José Luís Oliveira. 2018. Automated ICD-9-CM medical coding of diabetic patient's clinical reports. In Proceedings of the First International Conference on Data Science, E-learning and Information Systems (DATA '18). Association for Computing Machinery, New York, NY, USA, Article 23, 1–6. https://doi.org/10.1145/3279996.3280019</code> as implemented by Jaya Singh (jayas2) and Mukesh Naresh Chugani (chugani2) for CS 598: Deep Learning for Healthcare.

### Dependencies
Run the following command to install python dependencies
<pre><code>pip3 install -r final_codes/requirements.txt </code></pre>

### Data Download
The paper uses the MIMIC-III clinical dataset - a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients between 2001 and 2012. The access to this dataset can be requested in the [Physionet](https://physionet.org/content/mimiciii/1.4/) portal. We use the files <code>NOTEEVENTS.csv</code> and <code>DIAGNOSES_ICD.csv</code> for training and evaluating the model.

### Directory Structure
Two directories: final_codes (latest version of code files) and draft_codes (old version of files i.e before draft submission)

### Preprocessing
Load the pickle raw data of MIMIC-III to execute preprocessing steps and dump the resultant dataframe as a pickle file.
<pre><code>python final_codes/preprocess.py <path_to_raw_data_pickle></code></pre>

### Train/Evaluate
Load the preprocessed data from above step to start training/evaluation for one of the given nine models:

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

### Performance Evaluation
Following are the results obtained while training the model for 30 epochs using our implementation:

| Model      | Precision | Recall | F1-score |
| ----------- | ----------- | ----------- | ----------- |
| BoT Baseline      | 0.71 | 0.23 | 0.30 |
| BoT Baseline w/ Cat 0 Dense   | 0.71 | 0.23 | 0.30 |
| BoT Baseline w/ Cat 1 Dense   | 0.77 | 0.26 | 0.32 |
| BoT Baseline w/ Cat 2 Dense   | 0.85 | 0.25 | 0.33 |
| BoT Baseline w/ Cat 3 Dense   | 0.82 | 0.26 | 0.34 |
| CNN Baseline      | 0.62 | 0.36 | 0.42 |
| CNN Baseline w/ Cat 0 Dense   | 0.64 | 0.36 | 0.42 |
| CNN Baseline w/ Cat 1 Dense   | 0.48 | 0.34 | 0.38 |
| CNN Baseline w/ Cat 2 Dense   | 0.66 | 0.35 | 0.41 |
| CNN Baseline w/ Cat 3 Dense   | 0.66 | 0.35 | 0.41 |
| CNN 3-Conv1D   | **0.76** | **0.36** | **0.49** |
