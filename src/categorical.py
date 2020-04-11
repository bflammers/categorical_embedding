
import numpy as np
import pandas as pd

import json
import warnings

from src import io


def flip_signs(A, B):

    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """

    signs = np.sign(A) * np.sign(B)
    return A, B * signs


def pinv(X):

    """
    Moore-Penrose Pseudo Inverse
    This is almost the same implementation of the Moore-Penrose inverse as
    numpy.linalg.pinv only with a different handling of the smallest singular
    values. Better for anomaly detection purposes

    :param X: numpy.ndarray covariance matrix
    :return: numpy.ndarray inverted covariance matrix
    """

    assert len(X.shape) == 2, 'Only for covariance matrices!'

    # Perform singular value decomposition
    u, s, vt = np.linalg.svd(X, full_matrices=False)

    # Determine values that are too small as a divisor
    mask = s < (1e-15 * np.max(s))

    # Divide 1 by the singular values
    # for all singular values that are sufficiently large
    s_inv = np.divide(1, s, where=~mask)

    # Set the singular values that where too small to the maximum
    # --> this is the main difference with numpy.linalg.pinv
    s_inv[mask] = np.amax(s_inv)

    # To column vector
    s_inv = s_inv[..., np.newaxis]

    # Calculate pseudo inverse
    res = np.matmul(np.transpose(vt), np.multiply(s_inv, np.transpose(u)))

    return res


class PCA(object):

    """ PCA """

    def __init__(self):

        self._eig_vals = None
        self._eig_vecs = None

    def fit(self, X):

        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Calculate covariance matrix
        cov_mat = np.cov(X.T)

        # Calculate eigen values and eigen vectors
        self._eigen_decomposition(cov_mat)

    def _eigen_decomposition(self, X):

        # Eigen decomposition
        eig_vals, eig_vecs = np.linalg.eig(X)

        # Order by eigenvalues (decreasing)
        idx_order = np.argsort(-eig_vals)
        self._eig_vecs = np.real(eig_vecs[:, idx_order])
        self._eig_vals = np.real(eig_vals[idx_order])

    def transform(self, X, n_components=None):

        # Check fitted
        assert self._eig_vals is not None, 'Fitting step not performed'

        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Calculate components
        components = np.matmul(X, self._eig_vecs)

        # Select first n_components
        if n_components is not None:
            components = components[:, :n_components]

        # To DataFrame
        df_components = pd.DataFrame(components)
        df_components.columns = \
            ['CE_{}'.format(i) for i in range(df_components.shape[1])]

        return df_components

    def fit_transform(self, X, n_components=None):

        # Fitting step
        self.fit(X)

        # Return transformed data
        return self.transform(X, n_components)


class CategoricalEmbedding(PCA):

    """ A Categorical Embedding that preserves sparsity """

    def __init__(self, drop_zero_variance=True):

        super().__init__()
        self.drop_zero_variance = drop_zero_variance
        # self.cov_mat = None

    def write_json(self, fp):

        ser = {
            'drop_zero_variance': self.drop_zero_variance,
            '_eig_vals': self._eig_vals.tolist(),
            '_eig_vecs': self._eig_vecs.tolist()
        }

        with open(fp, 'w') as output_json:
            json.dump(ser, output_json, indent=4)

    def read_json(self, fp):

        with open(fp, 'r') as input_json:
            ser = json.load(input_json)

        self.drop_zero_variance = ser['drop_zero_variance']
        self._eig_vals = np.array(ser['_eig_vals'])
        self._eig_vecs = np.array(ser['_eig_vecs'])

    def fit(self, X):

        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Calculate covariance matrix
        cov_mat = np.cov(X.T)

        # Invert covariance matrix
        inv_cov_mat = pinv(cov_mat)

        # Calculate eigen values and eigen vectors
        self._eigen_decomposition(inv_cov_mat)

        # Remove components that do not capture any variance
        if self.drop_zero_variance:
            self._remove_zero_variance(X)

    def _remove_zero_variance(self, X):

        # TODO: this might be a bit dangerous - think about it some more

        """
        The logic behind this function is that when a couple of dimension in X
        are perfectly colinear, there are also some components that do not do
        anything and do not capture any variance. We want to remove these
        because they also do not capture rare events
        However, it is a bit tricky - where is the threshold?
        """

        # Calculate components
        components = super().transform(X, None)

        # Which components have a variance larger than the threshold
        components_mask = np.diag(np.cov(components.T)) > 1e-18

        n_zero_variance = np.sum(components_mask)
        if n_zero_variance > 5:
            message = '{} components zero variance! Does this make sense?!'\
                .format(n_zero_variance)
            warnings.warn(message)

        # Delete these components
        self._eig_vals = self._eig_vals[components_mask]
        self._eig_vecs = self._eig_vecs[:, components_mask]


class DummyEncoder(object):

    def __init__(self, schema):

        # Check schema is dict
        assert isinstance(schema, dict), 'Schema must be a dictionary'

        self.fields = schema
        self.dummy_names = list()

        for k, v in self.fields.items():
            # Checks on schema
            assert isinstance(k, str), 'Field name must be a string'
            assert len(k) > 0, 'Field name must have positive length'
            assert isinstance(v, list), 'Field categories must be a list'
            assert len(v) > 1, 'Field must have more than one categories'

            # Set a list with column names
            field_names = ['{}_{}'.format(k, cat) for cat in v]
            self.dummy_names += field_names

    def write_json(self, fp):

        ser = {
            '_fields': self.fields,
            '_dummy_names': self.dummy_names
        }

        io.write_json(ser, fp)

    def read_json(self, fp):

        self.dummy_names = list()

        ser = io.read_json(fp)

        self.fields = ser['_fields']
        self.dummy_names = ser['_dummy_names']

    def transform(self, X):

        """
        Encode categorical data into dummies
        :param X: The data to encoder
        :return: pd.DataFrame with dummies
        """

        # Convert to DataFrame
        df = io.to_DataFrame(X)

        # Split data in normal and categorical columns
        cat_cols = set(df.columns) & set(self.fields.keys())
        df_cat = df[cat_cols]
        df_num = df.drop(columns=cat_cols)

        # Get the dummies
        df_dum = pd.get_dummies(df_cat)

        # Check for any previously unseen categories
        if not set(df_dum.columns).issubset(self.dummy_names):

            not_in_set = set(df_dum.columns) - set(self.dummy_names)
            raise Exception('Categories {} not in dummy encoding schema!'
                            .format(not_in_set))

        # Set missing columns and in correct order
        df_dum = df_dum.reindex(columns=self.dummy_names)

        # Fill missing values
        df_dum.fillna(0, inplace=True)

        # Convert to int8 for better use of memory
        df_dum = df_dum.astype(np.int8)

        return df_num, df_dum


class BlackSwanDetector(object):

    """
    Class that detects black swans columns in binary data. Black swan columns
    are columns that contain only zeroes or only ones. So in the training set,
    we do not observe a single observation of the opposing class (black swans).
    We want to protect ourselves against these unobserved events.
    """

    def __init__(self, merge_prefixes=None):

        """
        Constructor method of class BlackSwanDetector.
        :param merge_prefixes: list of prefixes of column names that need to be
            merged together
        """

        self._merge_prefixes = [] if not merge_prefixes else merge_prefixes
        self._bs_columns = dict()

    def write_json(self, fp):

        ser = {
            '_merge_prefixes': self._merge_prefixes,
            '_bs_columns': self._bs_columns
        }

        with open(fp, 'w') as output_json:
            json.dump(ser, output_json, indent=4)

    def read_json(self, fp):

        with open(fp, 'r') as input_json:
            ser = json.load(input_json)

        self._merge_prefixes = ser['_merge_prefixes']
        self._bs_columns = ser['_bs_columns']

    def fit(self, X):

        """
        Fit method of class BlackSwanDetector detects if there are black swan
        columns in X and check if the column names in X match one of the
        prefixes in merge_prefixes. If there is a match, the black swan column
        will be merged with other matching columns
        :param df: pandas.DataFrame with data
        """

        # Convert to DataFrame
        df = io.to_DataFrame(X)

        # Loop over columns
        for col_name, values in df.iteritems():

            # Determine unique values in column
            uniq = np.unique(values)

            # If there is only one unique value, it is a black swan column
            if len(uniq) <= 1:

                value = uniq[0]
                assert np.equal(np.mod(value, 1), 0), \
                    'Black swan detector only for dummy variables'
                value = int(value)

                # Column details for this column
                column_details = {
                    'name': col_name,
                    'value': value
                }

                # Check if black swan column should be merged with others
                matched_prefix = [prefix for prefix in self._merge_prefixes
                                  if col_name.startswith(prefix + '_')]

                # Check if there is only one match per col_name
                assert len(matched_prefix) <= 1, \
                    'Name ({}) cannot match multiple prefixes ({})'\
                        .format(col_name, matched_prefix)

                # Check if there is a matched prefix
                if matched_prefix:

                    bs_name = '{}_black_swans'.format(matched_prefix[0])

                    # Check if the prefix was already matched by another column
                    if bs_name in self._bs_columns:

                        # Append the column details to this prefix
                        self._bs_columns[bs_name].append(column_details)

                    else:

                        # Add a new entry in the dict
                        self._bs_columns[bs_name] = [column_details, ]

                else:

                    # There is no matching prefix - base column name is used
                    self._bs_columns[col_name] = [column_details, ]

    def transform(self, X):

        # Convert to DataFrame
        df = io.to_DataFrame(X)

        # Initialize black swan DataFrame
        df_black_swans = pd.DataFrame(columns=self._bs_columns.keys())

        # Loop over the identified black swans columns
        for name, details in self._bs_columns.items():

            # Black swan details to pandas DataFrame
            df_details = io.to_DataFrame(details)
            df_details = df_details.set_index('name').transpose()

            # Check which columns match
            matching_cols = df.columns.intersection(df_details.columns)

            # Select columns
            df_select = df[matching_cols]

            # Construct numpy array of black swan values with same shape
            bs_values = np.tile(df_details.values, (df_select.shape[0], 1))

            # Compare --> not matching values are black swans
            bs_matrix = bs_values != df_select.values

            # Take the row sum to get the count of black swans per transaction
            # and add to DataFrame
            df_black_swans[name] = np.sum(bs_matrix, axis=1)

            # Drop the matching column
            df = df.drop(columns=matching_cols)

        return df, df_black_swans

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


if __name__ == '__main__':

    # Set up schema
    schema = {
        'method': ['credit_card', 'bank_transfer', 'ideal'],
        'country': ['NLD', 'BEL', 'GER', 'USA']
    }

    # Set input DataFrame
    df = pd.DataFrame({'method': ['credit_card', 'ideal', 'bank_transfer'],
                       'country': ['NLD', 'NLD', 'GER']})

    # Initialize dummy encoder
    dummy_encoder = DummyEncoder(schema)

    # Apply dummy encoder
    df_num, df_dum = dummy_encoder.transform(df)

    # Detect black swans
    black_swan_detector = BlackSwanDetector(['country'])

    # Fitting step
    black_swan_detector.fit(df_dum)

    df2 = pd.DataFrame({'method': ['credit_card', 'ideal', 'bank_transfer'],
                        'country': ['NLD', 'NLD', 'USA']})

    df_num_2, df_dum_2 = dummy_encoder.transform(df2)

    # Transform step
    X, df_black_swans = black_swan_detector.transform(df_dum_2)

    pd.set_option('display.max_columns', 500)
    print(X)
    print('black swans:\n', df_black_swans)







