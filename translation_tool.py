import os
import shutil
import time

from huaweicloud_sis.client.tts_client import TtsCustomizationClient
from huaweicloud_sis.bean.tts_request import TtsCustomRequest
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.exception.exceptions import ClientException

from huaweicloud_sis.client.asr_client import AsrCustomizationClient
from huaweicloud_sis.bean.asr_request import AsrCustomShortRequest
from huaweicloud_sis.bean.asr_request import AsrCustomLongRequest
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.bean.sis_config import SisConfig

import json
import tempfile

from cloud_config import *
import winsound


def generate_sound(text):
    pos_dir = tempfile.mkdtemp(dir='./data')
    pos_path = os.path.join(pos_dir, 'generate.wav')
    config = SisConfig()
    config.set_connect_timeout(5)
    config.set_read_timeout(10)
    ttsc_client = TtsCustomizationClient(app_key, app_secret, region,
                                         project_id, sis_config=config)

    ttsc_request = TtsCustomRequest(text)
    ttsc_request.set_property('english_cameal_common')
    ttsc_request.set_audio_format('wav')
    ttsc_request.set_sample_rate('8000')
    ttsc_request.set_volume(50)
    ttsc_request.set_pitch(0)
    ttsc_request.set_speed(0)
    ttsc_request.set_saved(True)
    ttsc_request.set_saved_path(pos_path)

    result = ttsc_client.get_ttsc_response(ttsc_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    winsound.PlaySound(pos_path, winsound.SND_ALIAS)

    shutil.rmtree(pos_path)


def get_sound_text(wav_path):
    path = wav_path
    path_audio_fromat = 'wav'
    path_property = 'chinese_8k_common'

    config = SisConfig()
    config.set_connect_timeout(5)
    config.set_read_timeout(10)
    asr_client = AsrCustomizationClient(app_key, app_secret, region,
                                        project_id, sis_config=config)

    data = io_utils.encode_file(path)
    asr_request = AsrCustomShortRequest(path_audio_fromat, path_property, data)
    asr_request.set_add_punc('yes')

    result = asr_client.get_short_response(asr_request)

    return result['result']['text']


def identify_picture(pic_path):
    return


def scan_file(file_path):
    return


def clear_data():
    for file in os.listdir('./data'):
        shutil.rmtree(file)


if __name__ == '__main__':
    generate_sound('hello, this is bill')

