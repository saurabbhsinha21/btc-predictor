async function getRecentPrices() {
  const url = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=60';
  try {
    const response = await fetch(url);
    const data = await response.json();
    return data.map(k => parseFloat(k[4])); // closing prices
  } catch (err) {
    console.error("Error fetching price data", err);
    return null;
  }
}

function linearRegression(prices) {
  const n = prices.length;
  const x = Array.from({ length: n }, (_, i) => i);
  const y = prices;

  const xMean = x.reduce((a, b) => a + b) / n;
  const yMean = y.reduce((a, b) => a + b) / n;

  const numerator = x.reduce((sum, xi, i) => sum + ((xi - xMean) * (y[i] - yMean)), 0);
  const denominator = x.reduce((sum, xi) => sum + ((xi - xMean) ** 2), 0);

  const slope = numerator / denominator;
  const intercept = yMean - slope * xMean;

  return { slope, intercept };
}

async function predict() {
  const targetPrice = parseFloat(document.getElementById("targetPrice").value);
  const targetTime = parseInt(document.getElementById("targetTime").value);
  const output = document.getElementById("output");

  if (!targetPrice || !targetTime) {
    output.textContent = "⚠️ Please enter both target price and target time.";
    return;
  }

  output.textContent = "⏳ Analyzing...";

  const prices = await getRecentPrices();
  if (!prices) {
    output.textContent = "❌ Failed to load price data.";
    return;
  }

  const { slope, intercept } = linearRegression(prices);
  const futureIndex = prices.length + targetTime; // Predict 'targetTime' minutes ahead
  const predictedPrice = slope * futureIndex + intercept;

  const difference = Math.abs(predictedPrice - targetPrice);
  const confidence = Math.min(95, Math.max(50, 80 - (difference / targetPrice) * 100));

  const prediction = predictedPrice > targetPrice ? "🔼 Above" : "🔽 Below";
  output.textContent = `📊 Predicted: ${prediction}\n📈 Predicted Price: $${predictedPrice.toFixed(2)}\n🎯 Confidence: ${confidence.toFixed(1)}%`;
}
