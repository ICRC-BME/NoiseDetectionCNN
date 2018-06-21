from noise_detector import noise_detector, mef3_channel_iterator
from timer import timer
import pymef



def example_0():
    """
    Predict probabilities for categories: noise,60hz,pathology and physiology based on given 3s long data segment (15000 samples)
    """
    # initialize detector instance
    detector = noise_detector()

    # load mef3 file
    session = './tests/test_signal.mefd'
    password = '***'
    info = pymef.read_ts_channel_basic_info(session_path=session,password=password)
    test_data = pymef.read_ts_channels_sample(session_path=session,password=password,channel_map=[info[0]['name']],sample_map=[0,15000])
    test_data = test_data[0].reshape(1,15000)

    # predict probabilities for given data segment
    yp = detector.predict(test_data)
    return yp


def example_1():
    """
    Predict probabilities for categories: noise,60hz,pathology and physiology for given channel
    Predict single example per iteration (minibatch_size = 1). Does not need big GPU memory but exhibits significantly higher computing time
    """
    # initialize detector instance
    detector = noise_detector()

    # load mef3 file
    session = './tests/test_signal.mefd'
    password = '***'
    info = pymef.read_ts_channel_basic_info(session_path=session, password=password)

    # initialize channel iterator instance
    mci = mef3_channel_iterator()

    # pre-loads data into mci buffer
    mci = mci.buffer(session=session,
                     password=password,
                     channel=[info[0]['name']],
                     sample_map=[0,info[0]['nsamp']])

    # set buffer options
    mci = mci.buffer_options(samples=15000, offset=5000, minibatch_size=1)

    yp = list()
    for k,data in enumerate(mci):
        yp.extend(detector.predict(data))

    return yp


def example_2():
    """
    Predict probabilities for categories: noise,60hz,pathology and physiology for given channel
    Predict multiple examples per iteration (minibatch_size > 1).
    Depends on GPU memory and speed. In general, should be slightly faster. -> not significant
    Do not use on CPU, it is slower then example_1.
    """
    # initialize detector instance
    detector = noise_detector()

    # load mef3 file
    session = './tests/test_signal.mefd'
    password = '***'
    info = pymef.read_ts_channel_basic_info(session_path=session, password=password)

    # initialize channel iterator instance
    mci = mef3_channel_iterator()

    # pre-loads data into mci buffer
    mci = mci.buffer(session=session,
                     password=password,
                     channel=[info[0]['name']],
                     sample_map=[0,info[0]['nsamp']])

    # set buffer options
    mci = mci.buffer_options(samples=15000, offset=5000, minibatch_size=100)

    yp = list()
    for k,data in enumerate(mci):
        yp.extend(detector.predict_minibatch(data))

    return yp

if __name__ == "__main__":
    with timer():
        y0 = example_0()

    with timer():
        y1 = example_1()

    with timer():
        y2 = example_2()





