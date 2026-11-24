import { useEffect, useState } from 'react'

const emotions = ['기쁨', '불안', '슬픔', '분노', '당황', '상처', '중립', '잘 모르겠어요']

export default function App() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState(null)
  const [selectedEmotion, setSelectedEmotion] = useState('')
  const [confirmedEmotion, setConfirmedEmotion] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [entries, setEntries] = useState([])
  const [historyRevision, setHistoryRevision] = useState(0)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [historyError, setHistoryError] = useState('')
  
  
  useEffect(() => {
    const controller = new AbortController()

    async function loadHistory() {
      setHistoryError('')

      try {
        const response = await fetch('/api/mood-entries', {
          signal: controller.signal,
        })

        if (!response.ok) throw new Error('기록 조회 실패')

        setEntries(await response.json())
      } catch {
        if (!controller.signal.aborted) {
          setHistoryError('최근 기록을 불러오지 못했어요.')
        }
      }
    }

    loadHistory()
    return () => controller.abort()
  }, [historyRevision])
  

  async function sendMessage(event) {
    event.preventDefault()

    const text = message.trim()
    if (!text || pending) return

    setPending(true)
    setError('')
    setResult(null)
    setSelectedEmotion('')
    setConfirmedEmotion('')

    try {
      const response = await fetch('/api/chat/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })

      if (!response.ok) {
        throw new Error(`요청 실패 (${response.status})`)
      }

      const data = await response.json()
      setResult({ ...data, message: text })
      setSelectedEmotion(data.suggestedEmotion)
      setMessage('')
    } catch {
      setError('답변을 불러오지 못했어요. 잠시 후 다시 시도해 주세요.')
    } finally {
      setPending(false)
    }
  }
  
  async function saveEmotion() {
    if (!selectedEmotion || saving || confirmedEmotion) return

    setSaving(true)
    setSaveError('')

    try {
      const response = await fetch('/api/mood-entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirmedEmotion: selectedEmotion }),
      })

      if (!response.ok) throw new Error('감정 저장 실패')

      setConfirmedEmotion(selectedEmotion)
      setHistoryRevision((revision) => revision + 1)
    } catch {
      setSaveError('감정을 저장하지 못했어요. 잠시 후 다시 시도해 주세요.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <main>
      <h1>감정 기록 챗봇</h1>

      <form onSubmit={sendMessage}>
        <label htmlFor="message">오늘 어떤 일이 있었나요?</label>
        <div>
          <textarea
            id="message"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="예: 내일 발표가 있어서 긴장돼."
            rows={4}
            maxLength={1000}
            disabled={pending}
          />
        </div>
        <button disabled={pending || !message.trim()}>
          {pending ? '답변을 불러오는 중…' : '보내기'}
        </button>
      </form>

      {error && <p role="alert">{error}</p>}

      {result && (
        <section aria-label="챗봇 답변">
          <p><strong>내가 보낸 문장</strong> {result.message}</p>
          <p><strong>챗봇</strong> {result.reply}</p>

          {result.mode === 'DEMO' && (
            <p>테스트 응답 · 예시 감정은 {result.suggestedEmotion}입니다.</p>
          )}

          <label htmlFor="emotion">내가 느낀 감정</label>{' '}
          <select
            id="emotion"
            value={selectedEmotion}
            onChange={(event) => {
              setSelectedEmotion(event.target.value)
              setConfirmedEmotion('')
            }}
          >
            {emotions.map((emotion) => (
              <option key={emotion} value={emotion}>{emotion}</option>
            ))}
          </select>{' '}
		  <button
		    type="button"
		    onClick={saveEmotion}
		    disabled={saving || !selectedEmotion || Boolean(confirmedEmotion)}
		  >
		    {saving ? '저장 중…' : '감정 기록 저장'}
		  </button>

		  {saveError && <p role="alert">{saveError}</p>}

		  {confirmedEmotion && (
		    <p role="status">{confirmedEmotion}을(를) 기록했어요.</p>
		  )}
        </section>
      )}
	  <section aria-labelledby="history-title">
	    <h2 id="history-title">최근 감정 기록</h2>

	    {historyError && <p role="alert">{historyError}</p>}

	    <ul>
	      {entries.map((entry) => (
	        <li key={entry.id}>
	          {entry.confirmedEmotion} ·{' '}
	          {new Date(entry.createdAt).toLocaleString('ko-KR')}
	        </li>
	      ))}
	    </ul>
	  </section>
    </main>
  )
}