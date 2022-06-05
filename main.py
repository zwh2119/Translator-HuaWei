import os
import time

from huaweicloud_sis.client.tts_client import TtsCustomizationClient
from huaweicloud_sis.bean.tts_request import TtsCustomRequest
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.exception.exceptions import ClientException

import json
import tempfile

from cloud_config import *
from winsound import PlaySound


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
    PlaySound(pos_path, flags=1)


if __name__ == '__main__':
    generate_sound('hello, this is bill')

