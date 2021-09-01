let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new SignDataset();
var zeroSamples = 0,
  oneSamples = 0,
  twoSamples = 0,
  threeSamples = 0,
  fourSamples = 0,
  fiveSamples = 0,
  sixSamples = 0,
  sevenSamples = 0,
  eightSamples = 0,
  nineSamples = 0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output
  });
}

async function predict() {
  model =await tf.loadLayersModel('https://sign-lang-website.herokuapp.com/my_model.json');
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch (classId) {
      case 0:
        predictionText = "I see Zero";
        break;
      case 1:
        predictionText = "I see One";
        break;
      case 2:
        predictionText = "I see Two";
        break;
      case 3:
        predictionText = "I see Three";
        break;
      case 4:
        predictionText = "I see Four";
        break;
      case 5:
          predictionText = "I see Five";
          break;
      case 6:
          predictionText = "I see Six";
          break;
      case 7:
          predictionText = "I see Seven";
          break;
      case 8:
          predictionText = "I see Eight";
          break;
      case 9:
          predictionText = "I see Nine";
          break;
    }
    document.getElementById("prediction").innerText = predictionText;


    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining() {
  train();
  alert("Training Done!")
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
  predict();
}


function saveModel() {
  model.save('downloads://my_model');
}


async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));

}


init();