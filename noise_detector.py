from model.model import neural_network
import torch
import numpy as np
from torch.autograd import Variable
from scipy import signal
from scipy import stats
import pymef

class noise_detector():
    def __init__(self,model_path,cuda_id = 0):
        # initialize new empty model
        self.net = neural_network()
        self.cuda_id = cuda_id
        # load system parameters from file
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        if torch.cuda.is_available():
            self.net = self.net.cuda(self.cuda_id)
            print('CUDA is available. Computations run on GPU: '+str(self.cuda_id))
        else:
            print('CUDA is not available. Computations run on CPU')


        self.bands=list()
        self.bands.append(signal.butter(N = 3, Wn = 900 / 2500,btype= 'low'))
        self.bands.append(signal.butter(N = 3, Wn = (20 / 2500, 100 / 2500), btype = 'bandpass'))
        self.bands.append(signal.butter(N = 3, Wn = (80 / 2500, 250 / 2500), btype = 'bandpass'))
        self.bands.append(signal.butter(N = 3, Wn = (200 / 2500, 600 / 2500), btype = 'bandpass'))
        self.bands.append(signal.butter(N = 3, Wn = (500 / 2500, 900 / 2500), btype = 'bandpass'))

        self.category = {0: 'noise', 1: 'ok', 2: 'patology'}

    def preprocessing(self,x):
        X = np.zeros((5, 15000))
        # LP 900Hz
        X[0,:] = signal.filtfilt(self.bands[0][0],self.bands[0][1],x)

        # BP Envelopes
        for i in range(1,5):
            X[i,:]=np.abs(signal.hilbert(signal.filtfilt(self.bands[i][0],self.bands[i][1],x)))

        # Z-score normalization
        X = stats.zscore(X,axis=1)

        X = Variable(torch.from_numpy(X).view(1,1,5,15000)).float()
        return X

    def predict(self,x):
        if type(x) is list:
            x=x[0]
        assert type(x) is np.ndarray
        assert x.shape == (1,15000)

        x = self.preprocessing(x)
        if torch.cuda.is_available():
            x = x.cuda()

        _,prob = self.net.forward(x)
        return self.output(prob.data.cpu().numpy())


    def predict_minibatch(self,x):
        assert type(x) is list
        data = list()
        for k in x:
            data.append(self.preprocessing(k))
        data = torch.cat(data,dim=0)

        if torch.cuda.is_available():
            data = data.cuda(self.cuda_id)

        _,prob = self.net.forward(data)
        return self.output(prob.data.cpu().numpy())


    def output(self,x):
        yy=list()
        for l in range(x.shape[0]):
            y = dict()
            for i,k in enumerate(self.category):
                y[self.category[k]] = x[l,i]
            yy.append(y)
        return yy


class mef3_channel_iterator():
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self.getMinibatch(self.buffer_minibatch_size)

    def getNext(self):
        if self.buffer_pos < self.buffer_data.shape[1]:
            x = self.buffer_data[:, self.buffer_pos : self.buffer_pos+self.buffer_samples]
            self.buffer_pos += self.buffer_offset
            if x.shape[1] != self.buffer_samples:
                raise StopIteration
            return x
        else:
            raise StopIteration

    def buffer_reset(self):
        self.buffer_pos = 0
        self.buffer_data = None
        return self

    def buffer_options(self,samples, offset, minibatch_size):
        self.buffer_offset = offset
        self.buffer_samples = samples
        self.buffer_minibatch_size = minibatch_size
        return self


    def buffer(self,session,password,channel,sample_map):
        assert type(session) is str
        assert type(channel) is list
        assert len(channel) is 1
        assert type(sample_map) is list
        assert len(sample_map) is 2


        self.buffer_reset()
        x = pymef.read_ts_channels_sample(session_path = session,
                                          password = password,
                                          channel_map = channel,
                                          sample_map = sample_map)
        self.buffer_data = np.array(x)
        return self


    def getMinibatch(self, batch_size):
        y = list()
        for k in range(batch_size):
            y.append(self.getNext())
        return y