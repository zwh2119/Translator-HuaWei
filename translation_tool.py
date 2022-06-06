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

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *

from image_sdk.utils import encode_to_base64
from image_sdk.image_tagging import image_tagging_aksk
from image_sdk.utils import init_global_env

import json
import tempfile

from cloud_config import *
import winsound

# from translate import *

import threading
import pyaudio
import wave

from huaweicloud_nlp.MtClient import MtClient
from huaweicloud_nlp.HWNlpClientAKSK import HWNlpClientAKSK

running = False


def generate_sound(text, is_en=True):
    pos_dir = tempfile.mkdtemp(dir=os.path.abspath('./data'))
    pos_path = os.path.join(pos_dir, 'generate.wav')
    config = SisConfig()
    config.set_connect_timeout(5)
    config.set_read_timeout(10)
    ttsc_client = TtsCustomizationClient(app_key, app_secret, region,
                                         project_id, sis_config=config)

    ttsc_request = TtsCustomRequest(text)
    if is_en:
        ttsc_request.set_property('english_cameal_common')
    else:
        ttsc_request.set_property('chinese_xiaoyan_common')
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


def make_audio():
    global running

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 16000  # Record at 44100 samples per second
    seconds = 60  # max seconds
    temp_dir = tempfile.mkdtemp(dir=os.path.abspath('./data'))
    filename = os.path.join(temp_dir, "output.wav")

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Start Recording..')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
        if not running:
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished Recording..')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    with open('data/out', 'w') as f:
        f.write(filename)
    print(filename)


def get_sound_text(wav_path):
    path = wav_path
    path_audio_fromat = 'wav'
    path_property = 'chinese_16k_common'

    config = SisConfig()
    config.set_connect_timeout(5)
    config.set_read_timeout(10)
    asr_client = AsrCustomizationClient(app_key, app_secret, region,
                                        project_id, sis_config=config)

    data = io_utils.encode_file(path)
    asr_request = AsrCustomShortRequest(path_audio_fromat, path_property, data)
    asr_request.set_add_punc('yes')

    result = asr_client.get_short_response(asr_request)
    print(result)

    return result['result']['text']


def identify_picture(pic_path):
    init_global_env('cn-north-4')
    result = image_tagging_aksk(app_key, app_secret, encode_to_base64(pic_path),
                                '', 'zh', 5, 60)

    result_dic = json.loads(result)
    print(result_dic)
    # file_name = os.path.split(pic_path)[1]

    res_list = result_dic['result']['tags']
    print(res_list)
    res_list = sorted(res_list, key=lambda x : float(x['confidence']), reverse=True)
    return_list = [i['i18n_tag'] for i in res_list]
    return return_list[:3]


def scan_file(file_path):
    credentials = BasicCredentials(app_key, app_secret)
    client = OcrClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(OcrRegion.value_of(region)) \
        .build()

    try:
        request = RecognizeGeneralTextRequest()

        request.body = {
            'image': encode_to_base64(file_path)
        }
        response = client.recognize_general_text(request)
        print(response)
        result = response.result.words_block_list
        print(result)
        res = ''
        for i in result:
            # print(i)
            res += i.words
        return res
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)
        raise Exception


def translate_cn_to_en(sentence):
    project_id = '854cdacd39be48858c5fa0abe8db0f61'
    akskClient = HWNlpClientAKSK(app_key,  # 用户的ak
                                 app_secret,  # 用户的sk
                                 "cn-north-4",  # region值
                                 project_id)  # projectId

    mtClient = MtClient(akskClient)
    response = mtClient.translate_text(sentence, "zh", "en", "common")
    result = response.res
    return result['translated_text']

    # return translate_text(sentence)

def translate_en_to_cn(sentence):
    project_id = '854cdacd39be48858c5fa0abe8db0f61'
    akskClient = HWNlpClientAKSK(app_key,  # 用户的ak
                                 app_secret,  # 用户的sk
                                 "cn-north-4",  # region值
                                 project_id)  # projectId

    mtClient = MtClient(akskClient)
    response = mtClient.translate_text(sentence, "en", "zh", "common")

    result = response.res
    return result['translated_text']


def search_pic(word):
    with open('image/img.json', 'r', encoding='utf8') as f:
        img_list = json.load(f)
    for i in img_list['image']:
        if i['en'] == word or i['zh'] == word:
            return i['path'], i['zh'], i['en']
    return '', '', ''


def clear_data():
    for file in os.listdir(os.path.abspath('./data')):
        shutil.rmtree(os.path.join(os.path.abspath('./data'), file))


if __name__ == '__main__':
    # print(translate_cn_to_en('我喜欢你'))
    # print(translate_cn_to_en('我真的很喜欢你'))
    # print(identify_picture('image/airplane.jpg'))
    # print(scan_file('C:\\Users\\15487\\Desktop\\test.jpg'))
    print(search_pic('plane'))
