/**
 * Send a chat message to the backend.
 *
 * @param {string} question - User question
 * @param {Array} history - Previous chat history
 * @param {string|null} conversationId - Optional conversation identifier
 */
export async function sendMessage(question, history = [], conversationId = null) {
    const res = await fetch('http://localhost:8000/query', { // <-- changed from '/chat'
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: question, history, conversation_id: conversationId })
    });
    return res.json();
}