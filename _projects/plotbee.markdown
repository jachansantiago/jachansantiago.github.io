---
layout: page
title: plotbee
description: Plotbee is a library to process, manage and visualize pose-based detections of bees.
image_small: /assets/img/plotbee/plotbee_small.webp
image_large: /assets/img/plotbee/plotbee_large.webp
github: https://github.com/jachansantiago/plotbee
importance: 1
---


<!-- Place this tag in your head or just before your close body tag. -->
<script async defer src="https://buttons.github.io/buttons.js"></script>

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/jachansantiago/plotbee" data-color-scheme="no-preference: light; light: light; dark: light;" data-size="large" aria-label="View on Github">View source on Github</a>

<!-- [Plotbee](https://github.com/jachansantiago/plotbee){:target="_blank"} -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lcppKrnbxGmJelXcuitNfclOW8_JdvEe?usp=sharing)



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <a href="https://github.com/jachansantiago/plotbee">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/plotbee.gif' | relative_url }}" alt="" title="Plotbee Image"/>
        </a>
    </div>
</div>
<div class="caption">
    Plotbee vizualization of Tracking.
</div>


## The Data Structure
Plotbee has a simple data structure based on videos, frames and bodies (skeleton detections). Where a video is a list of frames and a frame contains a list of bodies.

```python
from plotbee.video import Video
video = Video.load("plotbee_video_data.json")

for frame in video:
    for body in frame:
        pass
```


A body is consists as set of keypoints and connections. Plotbee functionalities are centered on video and body operations such as tracking (video operation) or pollen detection (body operation). Most of the visualization are at frame and body level. 

## Vizualization

The current plotbee format just store data related to keypoint detection, tracking information, pollen detection and tag detection. But do not store the image of the video. Therefore, for visualization the video file is need it.

```python
import videoplotter as vplt

video.load_video("plotbee_video.mp4")

frame = video[0]
body = frame[0]

vplt.imshow(frame)
vplt.imshow(body)
```

<!-- `vplt.imshow` has options to show skeletons, bounding boxes, tracking and id text.  -->



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <!-- <a href="https://github.com/jachansantiago/plotbee"> -->
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/plotbee_bbox_idtext.png' | relative_url }}" alt="" title="Plotbee Image"/>
        <!-- </a> -->
    </div>
</div>
<div class="caption">
    <code>vplt.imshow </code> has options to show skeletons, bounding boxes, tracking and id text. 
</div>

```python
vplt.imshow(frame, skeleton=False, bbox=True, tracks=False, idtext=True)
```

## Other Processes

#### Tracking
One of the important task to collect information about bees is tracking. The plotbee tracking is a greedy assigments of the detectecions based on distances.

```python
video.hungarian_tracking()
```

#### Pollen Detection
Pollen detection is based on a convolutional neural network to classify if a body contains pollen. If a body contains pollen then `body.pollen == True`.

```python
# Pollen Detection
video.process_pollen(model_path, weights=weights, workers=workers)

# Visualization
pollen_bees = list()

for frame in video:
    for body in frame:
        if body.pollen:
            pollen_bees.append(body)
            
shuffle(pollen_bees)
vplt.contact_sheet(pollen_bees[:20])
```

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <!-- <a href="https://github.com/jachansantiago/plotbee"> -->
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/pollen_bees.png' | relative_url }}" alt="" title="Plotbee Image"/>
        <!-- </a> -->
    </div>
</div>
<div class="caption">
    Contact sheet of bees detected with pollen. 
</div>


#### Tag Detection

In our setup, some individuals carries a marker to be recognize in long-term monitoring. More information about this tag setup in the [paper](https://doi.org/10.1145/3359115.3359120){:target="_blank"}. Plotbee includes detection of tags using an implemetation of [Apriltags](https://github.com/AprilRobotics/apriltag){:target="_blank"}. 

```python
# Tag Detection
video.tag_detection()

# Visualization
tagged_bees = video.tagged()

shuffle(tagged_bees)

vplt.tagged_contact_sheet(tagged_bees[:20])
```

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <!-- <a href="https://github.com/jachansantiago/plotbee"> -->
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/tagged_bees.png' | relative_url }}" alt="" title="Plotbee Image"/>
        <!-- </a> -->
    </div>
</div>
<div class="caption">
    Contact sheet of bees detected with tags. 
</div>

### Conclusion

This is just an overview of the plotbee library to show the power of this library, but there some features that are not explained here.

