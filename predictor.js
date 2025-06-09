let lastPrices = [];

async function fetchPrices() {
  const res = await fetch('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=500');
  const data = await res.json();
  return data.map(row => parseFloat(row[4])); // Close prices
}

function calculateEMA(prices, period) {
  const k = 2 / (period + 1);
  let ema = prices.slice(0, period).reduce((a, b) => a + b) / period;
  let result = [];
  for (let i = period; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
    result.push(ema);
  }
  return result;
}

function calculateRSI(prices, period = 14) {
  let rsi = [];
  for (let i = period; i < prices.length; i++) {
    let gains = 0, losses = 0;
    for (let j = i - period + 1; j <= i; j++) {
      const diff = prices[j] - prices[j - 1];
      if (diff >= 0) gains += diff;
      else losses -= diff;
    }
    const rs = gains / (losses || 1);
    rsi.push(100 - (100 / (1 + rs)));
  }
  return rsi;
}

function normalize(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  return arr.map(val => (val - min) / (max - min));
}

async function predict() {
  const targetPrice = parseFloat(document.getElementById("targetPrice").value);
  const targetTimeInput = document.getElementById("targetTime").value;
  const output = document.getElementById("output");

  if (!targetPrice || !targetTimeInput) {
    output.textContent = "⚠️ Please enter both target price and target time.";
    return;
  }

  const targetDate = new Date(targetTimeInput);
  const now = new Date();
  const minutesAhead = Math.floor((targetDate - now) / 60000);

  if (isNaN(targetDate.getTime()) || minutesAhead <= 0) {
    output.textContent = "❌ Invalid or past target time.";
    return;
  }

  output.textContent = "⏳ Fetching and training...";

  const prices = await fetchPrices();
  lastPrices = prices;

  const ema10 = calculateEMA(prices, 10);
  const ema50 = calculateEMA(prices, 50);
  const rsi = calculateRSI(prices);
  const returns = prices.slice(1).map((p, i) => p - prices[i]);

  // Trim arrays to the shortest length (due to EMA/RSI trimming)
  const minLen = Math.min(ema10.length, ema50.length, rsi.length, returns.length);
  const X = [];

  for (let i = 0; i < minLen; i++) {
    X.push([
      ema10[i],
      ema50[i],
      rsi[i],
      returns[i],
    ]);
  }

  // Normalize features
  const normalizedX = X[0].map((_, colIndex) => normalize(X.map(row => row[colIndex])));
  const inputs = normalizedX[0].map((_, rowIndex) => normalizedX.map(col => col[rowIndex]));

  const outputY = prices.slice(-minLen);

  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(outputY, [outputY.length, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 32, inputShape: [inputs[0].length], activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  await model.fit(xs, ys, { epochs: 50, verbose: 0 });

  const latestInput = inputs[inputs.length - 1];
  let predictedPrice = (await model.predict(tf.tensor2d([latestInput])).array())[0][0];

  // Simple extrapolation for future
  for (let i = 0; i < minutesAhead; i++) {
    const predTensor = model.predict(tf.tensor2d([latestInput]));
    predictedPrice = (await predTensor.array())[0][0];
  }

  const ema10Now = ema10.at(-1).toFixed(2);
  const ema50Now = ema50.at(-1).toFixed(2);
  const rsiNow = rsi.at(-1).toFixed(2);

  const diff = Math.abs(predictedPrice - targetPrice);
  const confidence = Math.min(95, Math.max(55, 85 - (diff / targetPrice) * 100));
  const direction = predictedPrice > targetPrice ? "🔼 Above" : "🔽 Below";

  output.innerHTML = `
    📊 Predicted: ${direction}<br>
    💰 Predicted Price: $${predictedPrice.toFixed(2)}<br>
    📈 EMA(10): $${ema10Now}, EMA(50): $${ema50Now}, RSI(14): ${rsiNow}<br>
    🎯 Confidence: ${confidence.toFixed(1)}%
  `;
}

// Optional auto-refresh
setInterval(() => {
  if (lastPrices.length > 0) fetchPrices().then(p => lastPrices = p);
}, 30000);
