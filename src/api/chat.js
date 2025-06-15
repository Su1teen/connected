export async function sendMessage(question, history = []) {
    const res = await fetch('http://localhost:8000/query', { // <-- changed from '/chat'
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: question, history })
    });
    return res.json();
}