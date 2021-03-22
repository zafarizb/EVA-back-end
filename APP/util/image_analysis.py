import os
# from ..fasterrcnn import fasterrcnn_image
from ..ssd import ssd_image
import config


conf = config.Config()


def run_image_analysis(taskname, username, filename, filepath, dnn):
    userpath = conf.FILE_PATH + username
    output_path = os.path.join(userpath, 'result', taskname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if dnn == 1:  # ssd
        instance = ssd_image.SsdImage(filename, filepath, output_path, 0, '')
        instance.run_ssd_image()
    else:
        pass
        # instance = fasterrcnn_image.FasterrcnnImage(filename, filepath, output_path, 0, '')
        # instance.run_fasterrcnn_image()

    return
