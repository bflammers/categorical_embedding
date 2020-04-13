
# Density Preserving Categorical Embedding

An embedding for categorical data that is well suited for anomaly detection. It involves a modified version of PCA that uses the inverse of the covariance matrix of binary encoded categories. The motivation for this embedding is currently mostly driven by heuristic thinking and experiments. It would be cool if someone with a more thorough computational statistics background could come up with a proper theory behind this approach! I would be happy to brainstorm.

![alt text][images/example.png]

## Intended outcome

* A reproducible transformation of categorical data to a small number (small relative to the number of categories, which is often very large) of features that can be used in an anomaly detection pipeline
* Categories and combinations of categories that are relatively rare (only a few occurences in training data) are mapped to a sparse area of the resulting low-dimensional feature space
* Categories and cobinations of categories that are normal (the bulk of the data) are mapped to the same area 
* Able to handle "black swans" (categories with no occurences in the training data)

## Method

PCA finds the projection onto a lower dimensional subspace that preserves as much variance as possible. When using PCA for dimensionality reduction this is exactly what you want. In anomaly detection we are interested in the exact opposite: strange or unusual variations. Things that do not occur often. 

There are multiple computational procedures to PCA. One involves an eigendecomposition of the covariance matrix of the dataset. We modify this procedure slightly and first compute the inverse of the covariance matrix, and then perform an eigendecomposition on the resulting matrix. This modification allows us to identify the directions in the data that capture the least covariance. 

By projecting the data onto these directions, we distinguish between rare observations and observations that that closely resember the bulk of the data. After projection, the bulk of the data is represented by one big lump, wheras the anomalues are well separated and located in sparse areas of the resulting low-dimensional space. This makes the transformation perfect as a pre-processing step in an anomaly detection pipeline.

In practice it often occurs that certain categories have no occurences in the training data - so called "black swan" events. We want our method to be robust for these events so that the pipeline does not break down in case such an event does occur during test time. We propose another pre-processing step that filters out these events and captures them in a separate feature so that the anomaly detection model can pick them up. 

**Train**
1. Dummy encode categorical training data &#8594; store schema "dummy_names"
2. Determine columns with all zero's (black swan columns) &#8594; store schema "bs_columns"
3. Remove columns in bs_schema from dummy encoded binary dataframe
4. Calculate covariance matrix
5. Calculate inverse of covariance matrix
6. Perform eigen decomposition of inverse covariance matrix &#8594; store "eig_vals" and "eig_vecs"

**Test**
1. Dummy encode categorical test data according to dummy_names schema
2. Separate black swan columns from dummy encoded binary dataframe
3. Check for ones in black swan columns and map found ones to separate dataframe (multiple ways to handle these)
4. Multiply remaining dummy encoded binary dataframe with eig_vals
5. Select the first components as the features to your anomaly detection model (by default the eigenvectors are ordered by eigenvalues in decreasing order) 

## Experiments

Covariance matrix of binary data

Exploratory visualizations

In practice: comparison to dummy encoding

In practice: comparison to other models

## Open questions

* What is the relation between the last p principal components in normal PCA and the first p components in this modified version of PCA? Are they equivalent?
* The Moore-Penrose pseudo inverse that is used in this method is calculated by a singular value decomposition (SVD), then taking the inverse of the diagonal singular value matrix, and then composing it all back together again. Since the SVD can also be used for PCA, is there a more computationally efficient method to arrive at the same transformation? This would allow a more efficient training step. It would not affect the test procedure. 

