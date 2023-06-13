const video1 = document.getElementsByClassName('input_video1')[0];
const out1 = document.getElementsByClassName('output1')[0];
const controlsElement1 = document.getElementsByClassName('control1')[0];
const canvasCtx1 = out1.getContext('2d');
const fpsControl = new FPS();

const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

function isImageBright(threshold, fuzz) {

  if(threshold === undefined) threshold = 127;
  if(fuzz === undefined || fuzz < 0 || fuzz % 2 !== 0) fuzz = 1;

  var data = canvasCtx1.getImageData(0, 0, out1.width, out1.height).data;
  var totalGrey = 0;

  for(var i = 0, ln = data.length, inc = 4 * fuzz; i < ln; i += inc) {
    var r = data[i],
        g = data[i+1],
        b = data[i+2],
        grey = Math.floor((r + g + b) / 3);

    totalGrey += grey;
  }

  var area = (out1.width * out1.height) / fuzz,
      lightLevel = Math.floor(totalGrey / area);

  return (lightLevel < threshold);
}


function isImageDark(image, threshold, fuzz) {

  if(threshold === undefined) threshold = 127;
  if(fuzz === undefined || fuzz < 0 || fuzz % 2 !== 0) fuzz = 1;

  var data = canvasCtx1.getImageData(0, 0, out1.width, out1.height).data;
  var totalGrey = 0;

  for(var i = 0, ln = data.length, inc = 4 * fuzz; i < ln; i += inc) {
    var r = data[i],
        g = data[i+1],
        b = data[i+2],
        grey = Math.floor((r + g + b) / 3);

    totalGrey += grey;
  }

  var area = (out1.width * out1.height) / fuzz,
      lightLevel = Math.floor(totalGrey / area);

  return (lightLevel < threshold);
}

function onResultsFace(results) {
  document.body.classList.add('loaded');
  fpsControl.tick();
  canvasCtx1.save();
  canvasCtx1.clearRect(0, 0, out1.width, out1.height);
  canvasCtx1.drawImage(results.image, 0, 0, out1.width, out1.height);

  canvasCtx1.font = "25px Arial";
  let text = null;
  if(results.detections.length > 1)
  {
    text = "한사람만 촬영하세요."
  }else if(results.detections.length === 0)
  {
    text = "정면 얼굴을 촬영해 주세요."
  } else if(results.detections.length === 1)
  {
    let widthRate = results.detections[0].boundingBox.width;
    let heightRate = results.detections[0].boundingBox.height

    if( widthRate > 0.8 || heightRate > 0.8)
    {
      text = "너무 가까워요.";
    }else if(widthRate < 0.6 || heightRate < 0.6)
    {
      text = "너무 멀어요.";
    }else if(isImageBright() ) {
      text = "너무 밝아요.";
    }else if(isImageDark() ) {
      text = "너무 어두워요.";
    }else {
      text = "적당해요";
    }
  }

  if(text != null)
  {
    let textWidth = canvasCtx1.measureText(text).width;
    canvasCtx1.fillText(text,out1.width*0.5-textWidth*0.5,out1.height - 40)
  }

    if (results.detections.length > 0) {
    drawRectangle(
        canvasCtx1, results.detections[0].boundingBox,
        {color: 'blue', lineWidth: 4, fillColor: '#00000000'});
    drawLandmarks(canvasCtx1, results.detections[0].landmarks, {
      color: 'red',
      radius: 5,
    });
  }

  canvasCtx1.restore();

}

const faceDetection = new FaceDetection({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.0/${file}`;
}});
faceDetection.onResults(onResultsFace);

const camera = new Camera(video1, {
  onFrame: async () => {
    await faceDetection.send({image: video1});
  },
  width: 480,
  height: 480
});
camera.start();

new ControlPanel(controlsElement1, {
      selfieMode: true,
      minDetectionConfidence: 0.5,
    })
    .add([
      new StaticText({title: 'MediaPipe Face Detection'}),
      fpsControl,
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01
      }),
    ])
    .on(options => {
      video1.classList.toggle('selfie', options.selfieMode);
      faceDetection.setOptions(options);
    });
