const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const cors = require("cors");
const { Firestore } = require("@google-cloud/firestore"); // Import Firestore

const app = express();
const port = 3000;

app.use(cors());

// Inisialisasi Google Cloud Firestore
const firestore = new Firestore({
  keyFilename: "credentials.json", // Ganti dengan path ke credentials Google Cloud Anda
});
const db = firestore; // Gunakan db untuk mengakses Firestore

// Setup Multer untuk menangani upload file
const storage = multer.memoryStorage(); // Menyimpan file dalam memori
const upload = multer({
  storage: storage,
  limits: { fileSize: 1000000 }, // Membatasi ukuran file menjadi 1MB
});

async function loadModel() {
    try {
      // Pastikan path model Anda benar
      model = await tf.loadGraphModel("file://submissions-model/model.json");
      console.log("Model loaded successfully");
    } catch (error) {
      console.error("Error loading model: ", error);
      throw new Error("Failed to load model.");
    }
  }
  
  loadModel(); // Memuat model saat aplikasi dimulai

// Endpoint untuk menerima gambar dan melakukan prediksi
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res
      .status(400)
      .json({ status: "fail", message: "No file uploaded" });
  }

  try {
    // Menangani error jika ukuran file lebih dari 1MB
    if (req.file.size > 1000000) {
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    }

    // Cek apakah file yang diupload adalah gambar
    if (!req.file.mimetype.startsWith("image/")) {
      return res
        .status(400)
        .json({ status: "fail", message: "Uploaded file is not an image" });
    }

    const imageBuffer = req.file.buffer;
    let imageTensor = tf.node.decodeImage(imageBuffer); // Decode image buffer

    // Resize image to 224x224 to match the model's input size
    imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);

    // Normalisasi gambar ke rentang [0, 1]
    imageTensor = imageTensor.div(255.0);

    // Memastikan tensor memiliki dimensi yang benar
    if (imageTensor.shape.length === 3) {
      imageTensor = imageTensor.expandDims(0); // Tambahkan batch dimension
    }

    // Lakukan prediksi menggunakan model
    const prediction = model.predict(imageTensor);
    const predictionData = prediction.dataSync();
    const predictedValue = parseFloat(predictionData[0].toFixed(3));

    // Prediksi pertama (misalnya untuk "Cancer")
    const result = predictedValue > 0.58 ? "Cancer" : "Non-cancer";
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";

    const resultData = {
      id: generateUUID(),
      result: result,
      suggestion: suggestion,
      createdAt: new Date().toISOString(),
    };

    // Simpan hasil prediksi ke Firestore
    await savePredictionToFirestore(resultData);

    // Kembalikan response ke pengguna
    res.status(201).json({
      status: "success",
      message: "Model is predicted successfully",
      data: resultData,
    });
  } catch (error) {
    console.error("Error during prediction: ", error);
    res.status(400).json({
      status: "fail",
      message: `Terjadi kesalahan dalam melakukan prediksi`,
    });
  }
});

// Endpoint untuk mendapatkan riwayat prediksi
app.get("/predict/histories", async (req, res) => {
  try {
    const predictionsRef = db.collection("predictions");
    const snapshot = await predictionsRef.get();

    const formattedPredictions = snapshot.docs.map((doc) => ({
      id: doc.id,
      history: doc.data(),
    }));

    res.json({
      status: "success",
      data: formattedPredictions,
    });
  } catch (error) {
    console.error("Error fetching histories:", error);
    res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan saat mengambil riwayat prediksi",
    });
  }
});

// Fungsi untuk menghasilkan UUID (ID unik untuk setiap prediksi)
function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Fungsi untuk menyimpan hasil prediksi ke Firestore
async function savePredictionToFirestore(resultData) {
  try {
    await db.collection("predictions").doc(resultData.id).set(resultData);
    console.log("Prediction saved to Firestore!");
  } catch (error) {
    console.error("Error saving prediction to Firestore:", error);
    throw new Error("Failed to save prediction.");
  }
}

// Menangani error file terlalu besar (lebih dari 1MB)
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  }
  next(err); // Lanjutkan ke error handler lainnya
});

// Jalankan server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
