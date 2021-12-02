
# base config

class config:
    # FPS while recording video from webcam.
    WEBCAM_FPS = 16#@param {type:"integer"}

    # Time in seconds to record video on webcam. 
    RECORDING_TIME_IN_SECONDS = 8. #@param {type:"number"}

    # Threshold to consider periodicity in entire video.
    THRESHOLD = 0.2#@param {type:"number"}

    # Threshold to consider periodicity for individual frames in video.
    WITHIN_PERIOD_THRESHOLD = 0.5#@param {type:"number"}

    # Use this setting for better results when it is 
    # known action is repeating at constant speed.
    CONSTANT_SPEED = False#@param {type:"boolean"}

    # Use median filtering in time to ignore noisy frames.
    MEDIAN_FILTER = True#@param {type:"boolean"}

    # Use this setting for better results when it is 
    # known the entire video is periodic/reapeating and
    # has no aperiodic frames.
    FULLY_PERIODIC = False#@param {type:"boolean"}

    # Plot score in visualization video.
    PLOT_SCORE = False#@param {type:"boolean"}

    # Visualization video's FPS.
    VIZ_FPS = 30#@param {type:"integer"}
    
    # Hummingbird flying.
    VIDEO_URL = 'https://imgur.com/t/hummingbird/m2e2Nfa'
    # VIDEO_URL = 'https://www.youtube.com/watch?v=5g1T-ff07kM'
    # VIDEO_URL = 'https://www.youtube.com/watch?v=5EYY2J3nb5c'