let lastPrices = [];

async function fetchPrices() {
  const res = await fetch('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100');
  const data = await res.json();
  return data.map(row => ({
    close: parseFloat(row[4]),
    volume: parseFloat(row[5])
  }));
}

function calculateEMA(values, period) {
  const k = 2 / (period + 1);
  let ema = values.slice(0, period).reduce((a, b) => a + b) / period;
  let result = [ema];
  for (let i = period; i < values.length; i++) {
    ema = values[i] * k + ema * (1 - k);
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
  return rsi;
}

function calculateMACD(prices, shortPeriod = 12, longPeriod = 26, signalPeriod = 9) {
  const shortEMA = calculateEMA(prices, shortPeriod).slice(-(prices.length - longPeriod));
  const longEMA = calculateEMA(prices, longPeriod);
  const macdLine = shortEMA.map((val, i) => val - longEMA[i + (shortPeriod - 1)]);
  const signalLine = calculateEMA(macdLine, signalPeriod);
  return {
    macd: macdLine,
    signal: signalLine,
    histogram: macdLine.map((v, i) => v - signalLine[i])
  };
}

function calculateStdDev(arr) {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length);
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
  if (minutesAhead <= 0) {
    output.textContent = "❌ Target time must be in the future.";
    return;
  }

  output.textContent = "⏳ Predicting...";
  const rawData = await fetchPrices();
  const prices = rawData.map(p => p.close);
  const volumes = rawData.map(p => p.volume);
  lastPrices = prices;

  // Calculate indicators
  const ema10 = calculateEMA(prices, 10);
  const rsi14 = prices.map((_, i) =>
    i >= 14 ? calculateRSI(prices.slice(i - 14, i + 1)) : 50
  );
  const macdData = calculateMACD(prices);
  const macd = macdData.macd;
  const signal = macdData.signal;

  // Build features
  const features = [];
  for (let i = 26; i < prices.length; i++) {
    features.push([
      prices[i],
      ema10[i - (10 - 1)] || prices[i],
      rsi14[i],
      macd[i - (26 - 1)] || 0,
      signal[i - (26 - 1)] || 0,
      volumes[i]
    ]);
  }

  const labels = prices.slice(26);

  const xs = tf.tensor2d(features);
  const ys = tf.tensor2d(labels.map(p => [p]));

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [6], units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  await model.fit(xs, ys, { epochs: 30, verbose: 0 });

  const lastFeature = features[features.length - 1];
  const predicted = model.predict(tf.tensor2d([lastFeature]));
  const predictedPrice = (await predicted.array())[0][0];

  const stdDev = calculateStdDev(prices);
  const diff = Math.abs(predictedPrice - targetPrice);
  const confidence = Math.max(30, Math.min(95, 90 - (diff / stdDev) * 10));
  const direction = predictedPrice > targetPrice ? "🔼 Above" : "🔽 Below";

  output.innerHTML = `
    📊 Predicted: ${direction} <br>
    💰 Predicted Price: $${predictedPrice.toFixed(2)} <br>
    📈 EMA(10): $${ema10.at(-1).toFixed(2)}, RSI(14): ${rsi14.at(-1).toFixed(2)} <br>
    📉 MACD: ${macd.at(-1).toFixed(2)}, Signal: ${signal.at(-1).toFixed(2)} <br>
    🎯 Confidence: ${confidence.toFixed(1)}%
  `;

  // Draw Chart
  const ctx = document.getElementById('chartCanvas').getContext('2d');
  if (window.priceChart) window.priceChart.destroy();

  window.priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: prices.map((_, i) => i),
      datasets: [
        {
          label: 'BTC Price',
          data: prices,
          borderColor: 'blue',
          tension: 0.2
        },
        {
          label: 'Predicted Price',
          data: Array(prices.length - 1).fill(null).concat(predictedPrice),
          borderColor: 'red',
          borderDash: [5, 5],
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        tooltip: { mode: 'index', intersect: false }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      },
      scales: {
        x: { display: true },
        y: { display: true }
      }
    }
  });
}
