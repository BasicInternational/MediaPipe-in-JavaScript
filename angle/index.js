import {
    FaceMesh,
    FACEMESH_TESSELATION,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_FACE_OVAL,
    FACEMESH_LIPS,
} from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors } from '@mediapipe/drawing_utils';
import { isMouthOpen, getFaceRotation } from './helpers';
import cv from '@mjyc/opencv.js';

const videoElement = document.querySelector('.input_video');
const canvasElement = document.querySelector('.output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const resultP = document.querySelector('p#result');

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(
        results.image,
        0,
        0,
        canvasElement.width,
        canvasElement.height
    );

    const landMarks = [];

    if (results.multiFaceLandmarks) {
        for (const face of results.multiFaceLandmarks) {
            face.forEach((landmarks) => {
                const { x, y, z } = landmarks;
                landMarks.push([x, y, z]);
            });

            drawConnectors(canvasCtx, face, FACEMESH_TESSELATION, {
                color: '#C0C0C070',
                lineWidth: 1,
            });
            drawConnectors(canvasCtx, face, FACEMESH_RIGHT_EYE, {
                color: '#FF3030',
            });
            drawConnectors(canvasCtx, face, FACEMESH_RIGHT_EYEBROW, {
                color: '#FF3030',
            });
            drawConnectors(canvasCtx, face, FACEMESH_RIGHT_IRIS, {
                color: '#FF3030',
            });
            drawConnectors(canvasCtx, face, FACEMESH_LEFT_EYE, {
                color: '#30FF30',
            });
            drawConnectors(canvasCtx, face, FACEMESH_LEFT_EYEBROW, {
                color: '#30FF30',
            });
            drawConnectors(canvasCtx, face, FACEMESH_LEFT_IRIS, {
                color: '#30FF30',
            });
            drawConnectors(canvasCtx, face, FACEMESH_FACE_OVAL, {
                color: '#E0E0E0',
            });
            drawConnectors(canvasCtx, face, FACEMESH_LIPS, { color: '#E0E0E0' });
        }
    }

    canvasCtx.restore();

    const image = cv.imread(canvasElement);
    // const isMouthOpen = check_landmarks_open(landMarks, image);

    const [pitch, yaw] = getFaceRotation(landMarks, image);
    const pitch_result = pitch < -10 ? 'down' : pitch > 30 ? 'up' : 'froward';
    const yaw_result = yaw < -30 ? 'right' : yaw > 30 ? 'left' : 'forward';

    resultP.textContent = JSON.stringify({ pitch_result, yaw_result }, null, 2);
}

const faceMesh = new FaceMesh({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    },
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
});

faceMesh.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await faceMesh.send({ image: videoElement });
    },
    width: 400,
    height: 300,
});

camera.start();

// Set the name of the hidden property and the change event for visibility
var hidden, visibilityChange;
if (typeof document.hidden !== 'undefined') {
    // Opera 12.10 and Firefox 18 and later support
    hidden = 'hidden';
    visibilityChange = 'visibilitychange';
} else if (typeof document.msHidden !== 'undefined') {
    hidden = 'msHidden';
    visibilityChange = 'msvisibilitychange';
} else if (typeof document.webkitHidden !== 'undefined') {
    hidden = 'webkitHidden';
    visibilityChange = 'webkitvisibilitychange';
}

function handleVisibilityChange() {
    if (document[hidden]) {
        camera.stop();
    } else {
        camera.start();
    }
}

// Warn if the browser doesn't support addEventListener or the Page Visibility API
if (typeof document.addEventListener === 'undefined' || hidden === undefined) {
} else {
    // Handle page visibility change
    document.addEventListener(visibilityChange, handleVisibilityChange, false);
}
