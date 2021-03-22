import os
import pathlib
import subprocess
# from ..fasterrcnn import fasterrcnn_image
from ..ssd import ssd_video_client, ssd_video_server
import config


conf = config.Config()


def run_video_analysis(taskname, username, dirpath, dnn, pl):
    userpath = conf.FILE_PATH + username
    input_path = pathlib.Path(dirpath)
    output_path = os.path.join(userpath, 'result', taskname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    client_ip = ('127.0.0.1', 10009)
    server_ip = ('127.0.0.1', 20009)

    if dnn == 1:  # ssd
        p = subprocess.Popen("D:\\Anaconda3\\Anaconda3\\envs\\py367\\python.exe "
                             "APP\\ssd\\video_server\\ssd_video_server.py " + dirpath + " " + output_path + " " + str(pl))

        instance = ssd_video_client.SsdVideoClient(input_path, output_path, pl, client_ip, server_ip)
        instance.run_ssd_video_client()

        p.wait()
    else:
        pass
        # instance = fasterrcnn_image.FasterrcnnImage(filename, filepath, output_path, 0, '')
        # instance.run_fasterrcnn_image()

    return


if __name__ == '__main__':
    dirpath = 'a'
    output_path = 'b'
    pl = 1
    p = subprocess.Popen("D:\\Anaconda3\\Anaconda3\\envs\\py367\\python.exe "
                         "APP\\ssd\\video_server\\ssd_video_server.py " + dirpath + " " + output_path + " " + str(pl))
