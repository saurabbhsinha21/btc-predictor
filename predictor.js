let lastPrices = [];

async function fetchPrices() {
  const res = await fetch('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100');
  const data = await res.json();
  return data.map(row => parseFloat(row[4]));
}

function calculateEMA(prices, period) {
  const k = 2 / (period + 1);
  let ema = prices.slice(0, period).reduce((a, b) => a + b) / period;
  let result = [ema];
  for (let i = period; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
    result.push(ema);
  }
  return result;
}

function calculateRSI(prices, period = 14) {
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = prices[i] - prices[i - 1];
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }
  let rs = gains / (losses || 1);
  let rsi = 100 - (100 / (1 + rs));
  return rsi.toFixed(2);
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

  if (isNaN(targetDate.getTime())) {
    output.textContent = "⚠️ Invalid date format.";
    return;
  }

  const minutesAhead = Math.floor((targetDate - now) / 60000);

  if (minutesAhead <= 0) {
    output.textContent = "❌ Target time must be in the future.";
    return;
  }

  output.textContent = "⏳ Predicting...";

  const prices = await fetchPrices();
  lastPrices = prices;

  const ema = calculateEMA(prices, 10).pop().toFixed(2);
  const rsi = calculateRSI(prices);

  const xs = tf.tensor(prices.map((_, i) => i), [prices.length, 1]);
  const ys = tf.tensor(prices, [prices.length, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 8, inputShape: [1], activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  await model.fit(xs, ys, { epochs: 20, verbose: 0 });

  const futureIndex = prices.length + minutesAhead;
  const predictedTensor = model.predict(tf.tensor([[futureIndex]]));
  const predictedPrice = (await predictedTensor.array())[0][0];

  const diff = Math.abs(predictedPrice - targetPrice);
  const confidence = Math.min(95, Math.max(55, 80 - (diff / targetPrice) * 100));
  const direction = predictedPrice > targetPrice ? "🔼 Above" : "🔽 Below";

  output.innerHTML = `
    📊 Predicted: ${direction} <br>
    💰 Predicted Price: $${predictedPrice.toFixed(2)} <br>
    📈 EMA(10): $${ema}, RSI(14): ${rsi} <br>
    🎯 Confidence: ${confidence.toFixed(1)}%
  `;
}

// Auto-refresh prices every 30 seconds
setInterval(() => {
  if (lastPrices.length > 0) {
    fetchPrices().then(p => lastPrices = p);
  }
}, 30000);
