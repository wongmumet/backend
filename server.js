const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

const app = express();
const port = 3000;

// Setup Multer to handle file upload and store in memory
const storage = multer.memoryStorage(); // Store file in memory
const upload = multer({
  storage: storage,
  limits: { fileSize: 1000000 }, // Limit file size to 1MB
});

// Function to load the TensorFlow model
let model;

async function loadModel() {
  try {
    model = await tf.loadGraphModel("file://submissions-model/model.json");
    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading model: ", error);
    throw new Error("Failed to load model.");
  }
}

loadModel(); // Load the model when the application starts

// Endpoint to accept image and make prediction
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res
      .status(400)
      .json({ status: "fail", message: "No file uploaded" });
  }

  try {
    // Multer already checks for file size, no need to check manually
    if (!req.file.mimetype.startsWith("image/")) {
      return res
        .status(400)
        .json({ status: "fail", message: "Uploaded file is not an image" });
    }

    const imageBuffer = req.file.buffer;
    let imageTensor = tf.node.decodeImage(imageBuffer); // Decode image buffer

    // Resize image to 224x224 to match the model's input size
    imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);

    // Convert to RGB if image has 4 channels (RGBA)
    if (imageTensor.shape[2] === 4) {
      imageTensor = imageTensor.slice([0, 0, 0], [-1, -1, 3]); // Keep first 3 channels (RGB)
    }

    // Normalize image to range [0, 1]
    imageTensor = imageTensor.div(255.0);

    // Ensure tensor has the correct shape: [batch_size, height, width, channels]
    if (imageTensor.shape.length === 3) {
      imageTensor = imageTensor.expandDims(0); // Add batch dimension
    }

    // Perform prediction using the model
    const prediction = model.predict(imageTensor);

    // Extract prediction result (assuming it's a binary classification)
    const predictionData = prediction.dataSync();
    const predictedValue = parseFloat(predictionData[0].toFixed(3));

    // Debugging: log the prediction data
    console.log("Prediction Data: ", predictedValue);

    // Determine result based on the predicted value (threshold set to 0.58)
    const result = predictedValue > 0.58 ? "Cancer" : "Non-cancer";

    // Suggestion based on the result
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";

    // Save prediction result to a local JSON file
    const resultData = {
      id: generateUUID(),
      result: result,
      suggestion: suggestion,
      createdAt: new Date().toISOString(),
    };

    // Save to a local JSON file (appends to existing file if it exists)
    const filePath = path.join(__dirname, "predictions.json");
    const predictions = fs.existsSync(filePath)
      ? JSON.parse(fs.readFileSync(filePath))
      : [];
    predictions.push(resultData);
    fs.writeFileSync(filePath, JSON.stringify(predictions, null, 2));

    // Respond with the prediction result
    return res.json({
      status: "success",
      message: "Model predicted successfully",
      data: resultData,
    });
  } catch (error) {
    console.error("Error during prediction: ", error);
    return res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

// Endpoint to retrieve prediction history
app.get("/predict/histories", async (req, res) => {
  try {
    const filePath = path.join(__dirname, "predictions.json");

    if (fs.existsSync(filePath)) {
      const predictions = JSON.parse(fs.readFileSync(filePath));

      // Format the history data as required
      const formattedPredictions = predictions.map((prediction) => ({
        id: prediction.id,
        history: {
          result: prediction.result,
          createdAt: prediction.createdAt,
          suggestion: prediction.suggestion,
          id: prediction.id,
        },
      }));

      return res.json({
        status: "success",
        data: formattedPredictions,
      });
    } else {
      return res.json({
        status: "success",
        data: [],
      });
    }
  } catch (error) {
    console.error("Error fetching histories:", error);
    return res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan saat mengambil riwayat prediksi",
    });
  }
});

// Helper function to generate UUID (unique ID for each prediction)
function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Error handling for file size exceeded
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  }
  next(err); // Pass on to the next error handler
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
