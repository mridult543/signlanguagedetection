let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
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

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(10);

  // In the space below create a neural network that can classify hand gestures
  // corresponding to rock, paper, scissors, lizard, and spock. The first layer
  // of your network should be a flatten layer that takes as input the output
  // from the pre-trained MobileNet model. Since we have 5 classes, your output
  // layer should have 5 units and a softmax activation function. You are free
  // to use as many hidden layers and neurons as you like.  
  // HINT: Take a look at the Rock-Paper-Scissors example. We also suggest
  // using ReLu activation functions where applicable.
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({
        units: 100,
        activation: "relu"
      }),
      tf.layers.dense({
        units: 10,
        activation: "softmax"
      })
    ]
  });


  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  // const optimizer = tf.train.adam(0.0001);// YOUR CODE HERE
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.0001)
  }); // YOUR CODE HERE);

  model.summary();
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
      }
    }
  });
}


function handleButton(elem) {
  switch (elem.id) {
    case "0":
      zeroSamples++;
      document.getElementById("zerosamples").innerText = "Samples for 0:" + zeroSamples;
      break;
    case "1":
      oneSamples++;
      document.getElementById("onesamples").innerText = "Samples for 1:" + oneSamples;
      break;
    case "2":
      twoSamples++;
      document.getElementById("twosamples").innerText = "Samples for 2:" + twoSamples;
      break;
    case "3":
      threeSamples++;
      document.getElementById("threesamples").innerText = "Samples for 3:" + threeSamples;
      break;
    case "4":
      fourSamples++;
      document.getElementById("foursamples").innerText = "Samples for 4:" + fourSamples;
      break;
    case "5":
      fiveSamples++;
      document.getElementById("fivesamples").innerText = "Samples for 5:" + fiveSamples;
      break;
    case "6":
      sixSamples++;
      document.getElementById("sixsamples").innerText = "Samples for 6:" + sixSamples;
      break;
    case "7":
      sevenSamples++;
      document.getElementById("sevensamples").innerText = "Samples for 7:" + sevenSamples;
      break;
    case "8":
      eightSamples++;
      document.getElementById("eightsamples").innerText = "Samples for 8:" + eightSamples;
      break;
    case "9":
      nineSamples++;
      document.getElementById("ninesamples").innerText = "Samples for 9:" + nineSamples;
      break;
  }
  label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
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