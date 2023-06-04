while [ 1 ] ; do python3 scraper.py; killall -9 chromium; killall -9 chromedriver; sleep 631; done;
