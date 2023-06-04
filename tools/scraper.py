import sys
import time
import json
import signal
import requests
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import os.path

SMALL_WAIT_TIME = 21
WAIT_TIME = 501

files = 0

def handler(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
    if res == 'y':
        driver.quit()
        exit(1)

signal.signal(signal.SIGINT, handler)

# def get_good_filename(video):
#     while True:
#         filename = f'crossroad_{str(video).zfill(6)}.mp4'
#         if not os.path.isfile(filename):
#             time.sleep(SMALL_WAIT_TIME)
#             print(f'Using {filename}')
#             return filename, video
#         else:
#             video += 1
    
    
def process_browser_log_entry(entry):
    response = json.loads(entry['message'])['message']
    return response

video = 0
timestamp = time.time()

terminate = False
while terminate is False:
    print('Video:', video)

    # Initialise selenium driver
    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument("--window-size=1920,1080")
    stream0 = 'http://krkvideo7.orionnet.online/cam1/embed.html?autoplay=true'
    stream1 = 'http://krkvideo2.orionnet.online/cam3188/embed.html?autoplay=true'
    stream2 = 'http://krkvideo14.orionnet.online/cam1560/embed.html?autoplay=true'
    driver = webdriver.Chrome(desired_capabilities=caps,options=options)
    driver.get(stream2)
    time.sleep(1)

    tries = 0
    while terminate is False:
        tries += 1
        if tries > 20:
            terminate = True
            break
        browser_log = driver.get_log('performance') 
        events = [process_browser_log_entry(entry) for entry in browser_log]
        events = [event for event in events if 'Network.response' in event['method']]
        time.sleep(1.1)

        iter = 0
        for e in events:
            iter += 1
            if time.time() - timestamp > WAIT_TIME:
                terminate = True
                break
            
            time.sleep(0.1)
            if 'response' not in e['params']:
                continue
            if e['params']['response']['url'].endswith('.ts'):
                url=e['params']['response']['url']
                r1 = requests.get(url, stream=True)
                if r1.status_code == 200:
                    timestamp = time.time()
                    # filename, video = get_good_filename(video)
                    now = datetime.now()
                    date = now.strftime("%d_%m_%Y_%H_%M_%S")
                    filename = f'crossroad_{date}.mp4'

                    files += 1
                    if files == 4:
                        terminate = True
                        break
                    tries =0
                    with open(filename, 'ab') as f:
                        data = b''
                        for chunk in r1.iter_content(chunk_size=1024):
                            if chunk:
                                data += chunk
                        f.write(data)
                else:
                    print("Received unexpected status code {}".format(r1.status_code))
                    terminate = True
                    break

        if time.time() - timestamp > WAIT_TIME:
            terminate = True
            break


