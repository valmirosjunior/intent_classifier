import h2o
import numpy as np
from h2o.automl import H2OAutoML

from src.core import file_manager
from src.ml import pipeline


class H2OHelper:
    def __init__(self, df_train, df_test) -> None:
        self.df_train = df_train
        self.df_test = df_test

        # Start the H2O cluster (locally)
        h2o.init()
        # Run AutoML for 20 base models
        self.aml = H2OAutoML(max_models=20, seed=1)

    def train(self):
        hf_train = h2o.H2OFrame(self.df_train)

        x = hf_train.columns
        y = 'label'
        x.remove(y)

        hf_train[y] = hf_train[y].asfactor()

        self.aml.train(x=x, y=y, training_frame=hf_train)

    def test(self):
        hf_test = h2o.H2OFrame(self.df_test)

        hf_preds = self.aml.predict(hf_test)

        preds = hf_preds.as_data_frame().predict.to_numpy()

        ## Improve
        y_test = self.df_test['label'].to_numpy()

        correct_predict = np.equal(preds, y_test).sum()

        accuracy = correct_predict / len(y_test)

        print('The accuracy of model was: ', accuracy)

    def get_leader_model(self):
        return self.aml.leader

    def show_leader_border(self):
        # View the AutoML Leaderboard
        lb = self.get_leader_model()

        return lb.head(rows=lb.nrows)

    def save_model(self, suffix_output_dir='classified_manual'):
        path = file_manager.filename_from_data_dir(f'output/h2o/models/{suffix_output_dir}')
        filename = self.get_leader_model().key

        model = self.aml.leader

        h2o.save_model(model=model, path=path, force=True, filename=filename)

        print(f'The model was saved at: {path}, with name: {filename}')

    def predict_labels(self, df_to_predict):
        model = self.get_leader_model()

        hf_preds = model.predict(h2o.H2OFrame(df_to_predict))

        predicted_labels = hf_preds.as_data_frame().predict.to_numpy()

        return predicted_labels
