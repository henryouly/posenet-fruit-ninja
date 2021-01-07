import * as posenet from '@tensorflow-models/posenet';
import {drawBoundingBox, drawSegment, drawKeypoints, drawSkeleton, isMobile, toggleLoadingUI} from './demo_util';

import {startModule, installPoseDragger} from 'fruit-ninja';

import dat from 'dat.gui';
import Stats from 'stats.js';

const videoWidth = 640; // screen.width;
const videoHeight = 480; // screen.height;
const miniVideoWidth = 160;
const miniVideoHeight = 120;
const miniScale = 0.25;
let actionCount = 0;
let lastPair = null;
let lastActionTime = 0;
const actionSpanTime = 100;
const miniDistance = 0;
const maxDistanceRatio = 0.3;
const maxDistance = 300; // maxDistanceRatio * videoWidth;
const leftRightMiniDistance = 10;
let clearCanvas = true;
let xOffset = 0; (screen.availWidth - videoWidth)/2;
let yOffset = 0; (screen.availHeight - videoHeight)/3;

const guiState = {
    // algorithm: 'multi-pose',
    algorithm: 'single-pose',
    input: {
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: {width: 200, height: 150},
        quantBytes: 2,
    },
    singlePoseDetection: {
        minPoseConfidence: 0.5,
        minPartConfidence: 0.1,
    },
    multiPoseDetection: {
        maxPoseDetections: 5,
        minPoseConfidence: 0.15,
        minPartConfidence: 0.1,
        nmsRadius: 30.0,
    },
    output: {
        showVideo: false,
        showSkeleton: true,
        showPoints: true,
        showBoundingBox: false,
    },
    net: null,
};

// Pose Dragger
let PoseDragger = function() {
    this.events = {};

    this.on = function(event, handler) {
        this.events[event] = handler;
    };

    this.fire = function(event, data) {
        if (!event in this.events) {
            throw new Error('');
        }
        this.events[event].apply(this, data);
    };
};

let PersonDragger = function() {
    this.left = new PoseDragger();
    this.right = new PoseDragger();
};

let personDragger1 = new PersonDragger();
let personDraggers = [personDragger1];

function showInfo(message) {
  let info = document.getElementById('info');
  info.textContent = message;
  info.style.display = 'block';
}

async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;

    const mobile = isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: mobile ? undefined : videoWidth,
            height: mobile ? undefined : videoHeight,
        },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();

    return video;
}

function getHand() {
    return 'right';
}

function detectPoseInRealTime(video, net) {
    let stats = new Stats();
    stats.showPanel(0);
    document.body.appendChild(stats.dom);
    const canvas = document.getElementById('output');
    const canvasMini = document.getElementById('output_mini');
    const ctx = canvas.getContext('2d');
    const ctxMini = canvasMini.getContext('2d');
    // since images are being fed from a webcam
    const flipHorizontal = true;

    canvas.width = videoWidth;
    canvas.height = videoHeight;
    canvasMini.width = miniVideoWidth;
    canvasMini.height = miniVideoHeight;

    async function poseDetectionFrame() {
        if (guiState.changeToArchitecture) {
            // Important to purge variables and free up GPU memory
            guiState.net.dispose();

            // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
            // version
            guiState.net = await posenet.load(+guiState.changeToArchitecture);

            guiState.changeToArchitecture = null;
        }

        // Begin monitoring code for frames per second
        stats.begin();

        // Scale an image down to a certain factor. Too large of an image will slow
        // down the GPU
        const imageScaleFactor = guiState.input.imageScaleFactor;
        const outputStride = +guiState.input.outputStride;

        let poses = [];
        let minPoseConfidence;
        let minPartConfidence;
        switch (guiState.algorithm) {
            case 'single-pose':
                const pose = await guiState.net.estimateSinglePose(
                    video, {flipHorizontal: true});
                poses.push(pose);

                minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
                minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
                break;
            case 'multi-pose':
                poses = await guiState.net.estimateMultiplePoses(
                    video, imageScaleFactor, flipHorizontal, outputStride,
                    guiState.multiPoseDetection.maxPoseDetections,
                    guiState.multiPoseDetection.minPartConfidence,
                    guiState.multiPoseDetection.nmsRadius);

                minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
                minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
                break;
        }

        // ctx.clearRect(0, 0, videoWidth, videoHeight);

        if (guiState.output.showVideo) {
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-videoWidth, 0);
            ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
            ctx.restore();
        }

        // For each pose (i.e. person) detected in an image, loop through the poses
        // and draw the resulting skeleton and keypoints if over certain confidence
        // scores
        poses.forEach(({
            score,
            keypoints,
        }) => {
            if (score >= minPoseConfidence) {
                const now = new Date().getTime();

                // if(actionCount % 20 == 0){
                //   ctx.clearRect(0, 0, videoWidth, videoHeight);
                //   lastKeypoint = null;
                // }

                // drawKeypoints(keypoints, minPartConfidence, ctx, scale = 1, radius = 3);

                // 小窗中绘制识别到的关键点
                ctxMini.clearRect(0, 0, canvasMini.width, canvasMini.height);
                drawKeypoints(keypoints, minPartConfidence, ctxMini, miniScale);
                drawSkeleton(keypoints, minPartConfidence, ctxMini, miniScale);

                if (guiState.output.showPoints) {
                    let leftWrist = keypoints.find((point) => (point.part === 'leftWrist' && point.score > minPartConfidence));
                    let rightWrist = keypoints.find((point) => (point.part === 'rightWrist' && point.score > minPartConfidence));
                    let currentPair = {left: leftWrist, right: rightWrist};
                    let filteredKeypoints = Object.values(currentPair).filter((el) => el !== undefined);
                    // console.log(JSON.stringify(filteredKeypoints));
                    if (filteredKeypoints.length > 0) {
                        // ctx.clearRect(0, 0, videoWidth, videoHeight);
                        if (lastPair == null) {
                            lastPair = currentPair;
                            lastActionTime = now;
                        } else {
                            if (actionCount >= 2) {
                                ctx.clearRect(0, 0, videoWidth, videoHeight);
                                actionCount = 0;
                                // personDragger1.left.fire("startDrag", []);
                            }

                            drawKeypoints(filteredKeypoints, minPartConfidence, ctx, 1, 10);

                            // left
                            if ('left' == getHand() && lastPair.left != null && currentPair.left != null) {
                                // let leftDistance = calcDistance(lastPair.left, currentPair.left);
                                // if (leftDistance > miniDistance && leftDistance < maxDistance) {
                                if (true) {
                                    let dx = currentPair.left.position.x - lastPair.left.position.x;
                                    let dy = currentPair.left.position.y - lastPair.left.position.y;
                                    let x = currentPair.left.position.x + xOffset;
                                    let y = currentPair.left.position.y + yOffset;

                                    // showInfo("x=" + x + ", y=" + y + ", distance=" + leftDistance);
                                    personDragger1.left.fire('returnValue', [dx, dy, x, y, null, 'left']);

                                    drawSegment([lastPair.left.position.y,
                                            lastPair.left.position.x,
                                        ],
                                        [currentPair.left.position.y,
                                            currentPair.left.position.x,
                                        ], 'black', 1, ctx);

                                    lastPair.left = currentPair.left;
                                    actionCount++;
                                }
                            }

                            // right
                            if ('right' == getHand() && lastPair.right != null && currentPair.right != null) {
                                // let rightDistance = calcDistance(lastPair.right, currentPair.right);
                                // if (rightDistance > miniDistance && rightDistance < maxDistance) {
                                if (true) {
                                    let dx = lastPair.right.position.x - currentPair.right.position.x;
                                    let dy = lastPair.right.position.y - currentPair.right.position.y;
                                    let x = lastPair.right.position.x + xOffset;
                                    let y = lastPair.right.position.y + yOffset;

                                    // showInfo("x=" + x + ", y=" + y);
                                    personDragger1.right.fire('returnValue', [dx, dy, x, y, null, 'right']);
                                    drawSegment([lastPair.right.position.y,
                                            lastPair.right.position.x,
                                        ],
                                        [currentPair.right.position.y,
                                            currentPair.right.position.x,
                                        ], 'DeepPink', 1, ctx);

                                    lastPair.right = currentPair.right;
                                    actionCount++;
                                }
                            }

                            lastActionTime = now;
                        }
                    }
                }
                if (guiState.output.showSkeleton) {
                    drawSkeleton(keypoints, minPartConfidence, ctx);
                }
                if (guiState.output.showBoundingBox) {
                    drawBoundingBox(keypoints, ctx);
                }
            }
        });

        // End monitoring code for frames per second
        stats.end();

        requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
}

function calcDistance(point1, point2) {
    if (point1.position.x <= 0 || point1.position.x >= videoWidth) {
        // console.log("x1=" + point1.position.x);
    }
    if (point2.position.x <= 0 || point2.position.x >= videoWidth) {
        // console.log("x2=" + point2.position.x);
    }
    if (point1.position.y <= 0 || point1.position.y >= videoHeight) {
        // console.log("y1=" + point1.position.y);
    }
    if (point2.position.y <= 0 || point2.position.y >= videoHeight) {
        // console.log("y2=" + point2.position.y);
    }
    diff = Math.sqrt(Math.pow(point1.position.x - point2.position.x, 2) +
        Math.pow(point1.position.y - point2.position.y, 2));
    return diff;
}

async function bindPage() {
    showInfo('Loading PoseNet model...');
    const net = await posenet.load(guiState.input);
    guiState.net = net;

    document.getElementById('main').style.display = 'block';
    document.getElementById('clear').onclick = function() {
        output = document.getElementById('output');
        ctx = output.getContext('2d');
        ctx.clearRect(0, 0, videoWidth, videoHeight);
    };

    let video;

    try {
      showInfo('Loading Video...');
      video = await loadVideo();
    } catch (e) {
      let info = document.getElementById('info');
      info.textContent = 'this browser does not support video capture,' +
          'or this device does not have a camera';
      info.style.display = 'block';
      throw e;
    }

    showInfo('Starting Game...');
    detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();

startModule();
installPoseDragger(personDragger1.right);
