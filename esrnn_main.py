from m4_data import prepare_m4_data
from ESRNN.utils_evaluation import evaluate_prediction_owa

from ESRNN import ESRNN

X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name='Yearly',
                                                               directory = './data',
                                                               num_obs=1000)

# Instantiate model
model = ESRNN(max_epochs=25, freq_of_test=5, batch_size=4, learning_rate=1e-4,
              per_series_lr_multip=0.8, lr_scheduler_step_size=10,
              lr_decay=0.1, gradient_clipping_threshold=50,
              rnn_weight_decay=0.0, level_variability_penalty=100,
              testing_percentile=50, training_percentile=50,
              ensemble=False, max_periods=25, seasonality=[],
              input_size=4, output_size=6,
              cell_type='LSTM', state_hsize=40,
              dilations=[[1], [6]], add_nl_layer=False,
              random_seed=1, device='cpu')

# Fit model
# If y_test_df is provided the model
# will evaluate predictions on
# this set every freq_test epochs
model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

# Predict on test set
y_hat_df = model.predict(X_test_df)

# Evaluate predictions
final_owa, final_mase, final_smape = evaluate_prediction_owa(y_hat_df, y_train_df,
                                                             X_test_df, y_test_df,
                                                             naive2_seasonality=1)