# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 23:15:58 2022

@author: Mehar Kalra
"""

import vlc
 
# importing time module
import time
 
# creating vlc media player object
media_player = vlc.MediaPlayer()
 
# media object
media = vlc.Media(r"Project1\kajari.mp4")
 
# setting media to the media player
media_player.set_media(media)
 
 
# start playing video
media_player.play()
 
# wait so the video can be played for 5 seconds
# irrespective for length of video
time.sleep(5)
 
# pausing the video
media_player.set_pause(1)