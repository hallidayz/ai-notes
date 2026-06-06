// Let's modify CalendarBackend to add a sleep to simulate network requests so we can measure the concurrent execution clearly
async function runBenchmark() {
  const url = 'http://localhost:3000/api/calendar/events';

  const connections = [
    { provider: 'apple', tokens: { user: 'u1', password: 'p1', url: 'http://apple.com/u1', access_token: 'fake_token' } },
    { provider: 'apple', tokens: { user: 'u2', password: 'p2', url: 'http://apple.com/u2', access_token: 'fake_token' } },
    { provider: 'apple', tokens: { user: 'u3', password: 'p3', url: 'http://apple.com/u3', access_token: 'fake_token' } },
    { provider: 'apple', tokens: { user: 'u4', password: 'p4', url: 'http://apple.com/u4', access_token: 'fake_token' } },
    { provider: 'apple', tokens: { user: 'u5', password: 'p5', url: 'http://apple.com/u5', access_token: 'fake_token' } },
  ];

  console.log("Starting benchmark...");
  const start = Date.now();
  try {
    const response = await fetch(url, {
      method: 'POST',
      body: JSON.stringify({ connections }),
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();
    const duration = Date.now() - start;
    console.log(`Time taken: ${duration}ms`);
    console.log(`Events returned: ${data.length}`);
  } catch (error) {
    const duration = Date.now() - start;
    console.log(`Time taken (with error): ${duration}ms`);
    console.error(error);
  }
}

runBenchmark();
