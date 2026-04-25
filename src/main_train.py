from train_SDNet import TrainSDNet
from train_SegNet import TrainSegNet
from train_ExtractNet import TrainExtractNet
from project_paths import prepared_dataset_dir

class MainTrain():
    def __init__(self, dataset='RHSEDB'):
        '''
        Select the dataset to train from ['CCSEDB', 'RHSEDB']
        '''
        self.dataset = dataset
        self.prepared_dataset_path = str(prepared_dataset_dir(dataset))
        self.train_sdnet = TrainSDNet(dataset=dataset)
        self.train_segnet = TrainSegNet(dataset=dataset)
        self.train_extractnet = TrainExtractNet(dataset=dataset)

    def train(self):
        # train SDNet
        self.train_sdnet.train_model(epochs=40, init_learning_rate=0.0001, batch_size=8)
        print('SDNet training has been completed')
        # get prior information and other data for SegNet and ExtractNet
        self.train_sdnet.calculate_prior_information_and_qualitative(save_path=self.prepared_dataset_path)
        print('calculating prior information has been completed')
        # train SegNet
        self.train_segnet.train_model(epochs=10, init_learning_rate=0.0001, batch_size=8, dataset_path=self.prepared_dataset_path)
        print('SegNet training has been completed')
        #train ExtractNet
        self.train_extractnet.train_model(epochs=20, init_learning_rate=0.0001, batch_size=8, dataset=self.prepared_dataset_path)
        print('ExtractNet training has been completed')


if __name__ == '__main__':
    train_ = MainTrain(dataset='RHSEDB')
    train_.train()
