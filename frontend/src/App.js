import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [responseData, setResponseData] = useState(null);
  const [processSteps, setProcessSteps] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setAnswer('');
    setResponseData(null);
    setProcessSteps([]);

    // Simulated process steps aligned with current backend behavior
    const steps = [
      { step: 1, message: '🔍 Analyzing question and determining search strategy...', status: 'processing' },
      { step: 2, message: '📚 Searching facts database...', status: 'processing' },
      { step: 3, message: '🧪 Assessing adequacy of facts-based answer...', status: 'processing' },
      { step: 4, message: '🤖 Generating final answer...', status: 'processing' }
    ];
    setProcessSteps(steps);

    try {
      const response = await axios.post('/ask', {
        question: question
      });

      setResponseData(response.data);
      setAnswer(response.data.answer);
      
      // Update process steps based on response
      const finalSteps = [...steps];
      if (response.data.status === 'refused') {
        finalSteps[0] = { ...finalSteps[0], status: 'completed', message: '🚫 Question refused - sensitive topic detected' };
      } else if (response.data.status === 'answered') {
        finalSteps[0] = { ...finalSteps[0], status: 'completed', message: '✅ Question analyzed - proceeding with search' };
        
        // Check sources used
        const hasFacts = response.data.citations.some(c => c.source === 'byd_seal_facts.md');
        const hasExternal = response.data.citations.some(c => c.source === 'byd_seal_external.json');
        
        // Facts-only path: adequacy passed
        if (hasFacts && !hasExternal) {
          finalSteps[1] = { ...finalSteps[1], status: 'completed', message: '✅ Facts database searched' };
          finalSteps[2] = { ...finalSteps[2], status: 'completed', message: '✅ Adequacy passed — facts are sufficient' };
          finalSteps[3] = { ...finalSteps[3], status: 'completed', message: '✅ Answer generated from facts' };
        }

        // External path: facts inadequate or no facts → switch to external
        if (hasExternal) {
          finalSteps[1] = { ...finalSteps[1], status: 'completed', message: hasFacts ? '✅ Facts database searched' : '✅ Facts database searched (no relevant facts found)' };
          finalSteps[2] = { ...finalSteps[2], status: 'completed', message: '⚠️ Adequacy failed — switching to external reviews' };
          // Insert explicit external search step before final generation, if not already present
          finalSteps.splice(3, 0, { step: 3.5, message: '🌐 External database searched', status: 'completed' });
          finalSteps[4] = { ...finalSteps[4], status: 'completed', message: hasFacts ? '✅ Answer generated using external + facts context' : '✅ Answer generated from external reviews' };
        }
      } else {
        finalSteps[finalSteps.length - 1] = { ...finalSteps[finalSteps.length - 1], status: 'completed', message: '❌ No relevant information found' };
      }
      setProcessSteps(finalSteps);
      
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while processing your question.');
      setProcessSteps(steps.map(step => ({ ...step, status: 'error', message: '❌ Error occurred' })));
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'answered':
        return '#28a745';
      case 'refused':
        return '#dc3545';
      case 'no_information_found':
        return '#ffc107';
      default:
        return '#6c757d';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'answered':
        return 'Answered';
      case 'refused':
        return 'Refused';
      case 'no_information_found':
        return 'No Information Found';
      default:
        return status;
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 RAG Pipeline - BYD Seal Q&A</h1>
        <p>Ask questions about the BYD Seal and get AI-powered answers</p>
      </header>

      <main className="App-main">
        <div className="question-form">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="question">Ask a question:</label>
              <textarea
                id="question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g., What is the battery capacity? What colors are available?"
                rows="3"
                disabled={loading}
              />
            </div>
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? 'Processing...' : 'Ask Question'}
            </button>
          </form>
        </div>

        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {processSteps.length > 0 && (
          <div className="process-steps">
            <h3>🔄 RAG Pipeline Process</h3>
            <div className="steps-container">
              {processSteps.map((step, index) => (
                <div key={index} className={`step ${step.status}`}>
                  <div className="step-number">{step.step}</div>
                  <div className="step-message">{step.message}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {answer && (
          <div className="answer-section">
            <div className="answer-header">
              <h3>💡 Answer</h3>
              {responseData && (
                <div className="status-badge" style={{ backgroundColor: getStatusColor(responseData.status) }}>
                  {getStatusText(responseData.status)}
                </div>
              )}
            </div>
            
            <div className="answer-content">
              <p>{answer}</p>
              
              {/* Source indicator */}
              {responseData && responseData.citations && responseData.citations.length > 0 && (
                <div className="source-indicator">
                  <div className="source-badges">
                    {responseData.citations.some(c => c.source === 'byd_seal_facts.md') && (
                      <span className="source-badge facts">📋 Facts Database</span>
                    )}
                    {responseData.citations.some(c => c.source === 'byd_seal_external.json') && (
                      <span className="source-badge external">🌐 External Reviews</span>
                    )}
                  </div>
                  {responseData.citations.some(c => c.source === 'byd_seal_facts.md') && 
                   responseData.citations.some(c => c.source === 'byd_seal_external.json') && (
                    <div className="transition-note">
                      <small>💡 Answer combines facts with external reviews for comprehensive information</small>
                    </div>
                  )}
                </div>
              )}
            </div>

            {responseData && (
              <div className="response-details">
                <div className="detail-grid">
                  <div className="detail-item">
                    <strong>Status:</strong> {getStatusText(responseData.status)}
                  </div>
                  <div className="detail-item">
                    <strong>Citations:</strong> {responseData.citations?.length || 0}
                  </div>
                  <div className="detail-item">
                    <strong>Sources:</strong> {responseData.citations?.map(c => c.source).filter((v, i, a) => a.indexOf(v) === i).join(', ') || 'None'}
                  </div>
                </div>

                {responseData.citations && responseData.citations.length > 0 && (
                  <div className="citations">
                    <h4>📚 Citations</h4>
                    <div className="citations-grid">
                      {responseData.citations.map((citation, index) => (
                        <div key={index} className="citation-item">
                          <div className="citation-source">
                            <span className={`source-badge ${citation.type || citation.source}`}>
                              {citation.type === 'external_review' ? '🎥 Review' : '📋 Facts'}
                            </span>
                          </div>
                          <div className="citation-details">
                            {citation.type === 'external_review' ? (
                              <>
                                <div><strong>Title:</strong> {citation.title || 'N/A'}</div>
                                <div><strong>Channel:</strong> {citation.channel || 'N/A'}</div>
                                <div><strong>Views:</strong> {citation.views || 'N/A'}</div>
                                <div><strong>Subscribers:</strong> {citation.subscribers || 'N/A'}</div>
                              </>
                            ) : (
                              <>
                                <div><strong>Document:</strong> {citation.doc_id}</div>
                                <div><strong>Chunk:</strong> {citation.chunk_id}</div>
                              </>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
