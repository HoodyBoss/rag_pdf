'use client'

import { useState, useEffect } from 'react'
import { UserIcon, DocumentTextIcon, ChatBubbleLeftRightIcon, ArrowRightStartOnRectangleIcon, CloudArrowUpIcon, TrashIcon, ClipboardDocumentIcon, StarIcon, ChartBarIcon, CogIcon, CpuChipIcon, ShareIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline'
import axios from 'axios'

// Configure axios to point to the backend
const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
})

interface User {
  id: string
  username: string
  full_name?: string
}

interface ChatMessage {
  id: string
  user_message: string
  bot_response: string
  timestamp: string
  session_id: string
  feedback?: number
}

interface GeneralChatMessage {
  id: string
  sender: 'user' | 'bot'
  message: string
  timestamp: string
}

interface Document {
  id: string
  filename: string
  upload_time: string
  file_size: number
  chat_count: number
  is_favorite: boolean
}

export default function HomePage() {
  const [user, setUser] = useState<User | null>(null)
  const [documents, setDocuments] = useState<Document[]>([])
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [currentMessage, setCurrentMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // General Chat State
  const [generalChatMessages, setGeneralChatMessages] = useState<GeneralChatMessage[]>([])
  const [generalChatInput, setGeneralChatInput] = useState('')
  const [isGeneralChatLoading, setIsGeneralChatLoading] = useState(false)
  const [showMemoryModal, setShowMemoryModal] = useState(false)
  const [showSocialBotsModal, setShowSocialBotsModal] = useState(false)
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [loginUsername, setLoginUsername] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [loginError, setLoginError] = useState('')

  // AI Providers State
  const [availableProviders, setAvailableProviders] = useState<string[]>([])
  const [selectedProvider, setSelectedProvider] = useState('openai')
  const [providerModels, setProviderModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({})
  const [showApiKeyInput, setShowApiKeyInput] = useState(false)
  const [newApiKey, setNewApiKey] = useState('')

  // Google Sheets State
  const [sheetsUrl, setSheetsUrl] = useState('')
  const [showSheetsModal, setShowSheetsModal] = useState(false)
  const [sheetsPreview, setSheetsPreview] = useState<any>(null)
  const [showAIConfigModal, setShowAIConfigModal] = useState(false)
  const [sheetsProcessingStatus, setSheetsProcessingStatus] = useState<string>('')

  // Upload Enhancement State
  const [uploadType, setUploadType] = useState<'pdf' | 'sheets'>('pdf')
  const [isProcessing, setIsProcessing] = useState(false)

  // Memory Management State
  const [memoryStats, setMemoryStats] = useState<any>(null)
  const [newMemoryType, setNewMemoryType] = useState('working')
  const [newMemoryContent, setNewMemoryContent] = useState('')
  const [memorySearchQuery, setMemorySearchQuery] = useState('')
  const [memorySearchResults, setMemorySearchResults] = useState<any[]>([])

  // Social Bots State
  const [botConfigs, setBotConfigs] = useState<any>({})
  const [newBotPlatform, setNewBotPlatform] = useState('discord')
  const [newBotToken, setNewBotToken] = useState('')
  const [newBotWebhook, setNewBotWebhook] = useState('')
  const [newBotChannels, setNewBotChannels] = useState('')
  const [selectedBotPlatform, setSelectedBotPlatform] = useState('discord')
  const [selectedBotChannel, setSelectedBotChannel] = useState('')
  const [botMessage, setBotMessage] = useState('')

  // Feedback State
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)
  const [feedbackRating, setFeedbackRating] = useState(0)
  const [feedbackType, setFeedbackType] = useState('good')
  const [feedbackText, setFeedbackText] = useState('')
  const [selectedChatForFeedback, setSelectedChatForFeedback] = useState<any>(null)
  const [feedbackStats, setFeedbackStats] = useState<any>(null)

  useEffect(() => {
    checkAuth()
    loadDocuments()
    loadAIProviders()
    loadApiKeys()
    fetchFeedbackStats()
  }, [])

  const checkAuth = async () => {
    try {
      const response = await api.get('/api/auth/me')
      setUser(response.data)
    } catch (error) {
      setUser(null)
    }
  }

  const loadDocuments = async () => {
    try {
      const response = await api.get('/api/documents')
      setDocuments(response.data.documents || [])
    } catch (error) {
      console.error('Failed to load documents:', error)
    }
  }

  const handleLogin = async () => {
    try {
      setLoginError('')
      const response = await api.post('/api/auth/login', {
        username: loginUsername,
        password: loginPassword
      })

      if (response.data.success) {
        setUser(response.data.user)
        setShowLoginModal(false)
        setLoginUsername('')
        setLoginPassword('')
        setLoginError('')
      }
    } catch (error: any) {
      setLoginError(error.response?.data?.detail || 'Login failed. Please try again.')
      console.error('Login failed:', error)
    }
  }

  const handleLogout = async () => {
    try {
      // Simple client-side logout - no server endpoint needed
      setUser(null)
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }

  // AI Providers Functions
  const loadAIProviders = async () => {
    try {
      const response = await api.get('/api/ai/providers')
      const providers = response.data.available || []
      setAvailableProviders(providers)
      if (providers.length > 0) {
        setSelectedProvider(providers[0])
        loadProviderModels(providers[0])
      }
    } catch (error) {
      console.error('Failed to load AI providers:', error)
    }
  }

  const loadProviderModels = async (provider: string) => {
    try {
      const response = await api.get(`/api/ai/providers/${provider}/models`)
      setProviderModels(response.data.models || [])
      if (response.data.models?.length > 0) {
        setSelectedModel(response.data.models[0])
      }
    } catch (error) {
      console.error('Failed to load provider models:', error)
    }
  }

  const handleProviderChange = (provider: string) => {
    setSelectedProvider(provider)
    loadProviderModels(provider)
  }

  // API Key Management Functions
  const loadApiKeys = async () => {
    try {
      const response = await api.get('/api/config/api-keys')
      setApiKeys(response.data.api_keys || {})
    } catch (error) {
      console.error('Failed to load API keys:', error)
    }
  }

  const saveApiKey = async () => {
    if (!newApiKey.trim() || !selectedProvider) return

    try {
      await api.post('/api/config/api-key', {
        provider: selectedProvider,
        api_key: newApiKey.trim()
      })
      setApiKeys(prev => ({ ...prev, [selectedProvider]: newApiKey.trim() }))
      setNewApiKey('')
      setShowApiKeyInput(false)
    } catch (error) {
      console.error('Failed to save API key:', error)
    }
  }

  const deleteApiKey = async (provider: string) => {
    try {
      await api.delete(`/api/config/api-key/${provider}`)
      setApiKeys(prev => {
        const newKeys = { ...prev }
        delete newKeys[provider]
        return newKeys
      })
    } catch (error) {
      console.error('Failed to delete API key:', error)
    }
  }

  // Google Sheets Functions
  const previewGoogleSheets = async () => {
    if (!sheetsUrl.trim()) return

    try {
      setIsProcessing(true)
      const response = await api.post('/api/google-sheets/preview', {
        sheets_url: sheetsUrl
      })
      setSheetsPreview(response.data)
    } catch (error) {
      console.error('Failed to preview Google Sheets:', error)
      if (error.response?.status === 500) {
        alert('Google Sheets preview is currently unavailable. Vector database is not configured.')
      } else {
        alert('Failed to preview Google Sheets. Please check the URL and try again.')
      }
    } finally {
      setIsProcessing(false)
    }
  }

  const processGoogleSheets = async () => {
    if (!sheetsUrl.trim()) return

    try {
      setIsProcessing(true)
      setSheetsProcessingStatus('กำลังเชื่อมต่อกับ Google Sheets...')

      const response = await api.post('/api/google-sheets/process', {
        sheets_url: sheetsUrl
      })

      setSheetsProcessingStatus('กำลังประมวลผลข้อมูล...')

      const result = response.data

      if (result.success) {
        // Show detailed success message
        const chunksProcessed = result.chunks_processed || 0
        const sheetsData = result.sheets_data || {}
        const rowCount = sheetsData.row_count || 0
        const columnCount = sheetsData.column_count || 0
        const sheetName = sheetsData.sheet_name || 'Unknown'

        setSheetsProcessingStatus(`สำเร็จ! ประมวลผล ${chunksProcessed} ชิ้นส่วนจาก ${rowCount} แถว`)

        const successMessage = `✅ ประมวลผล Google Sheets สำเร็จ!\n\n` +
          `📊 Sheet: ${sheetName}\n` +
          `📈 ข้อมูลทั้งหมด: ${rowCount} แถว × ${columnCount} คอลัมน์\n` +
          `🔧 จำนวนที่ process แล้ว: ${chunksProcessed} ชิ้นส่วน\n` +
          `🔗 URL: ${sheetsUrl}`

        alert(successMessage)

        loadDocuments()
        setShowSheetsModal(false)
        setSheetsUrl('')
        setSheetsPreview(null)
      } else {
        setSheetsProcessingStatus(`ล้มเหลว: ${result.message || 'Unknown error'}`)
        alert(`❌ ประมวลผล Google Sheets ล้มเหลว: ${result.message || 'Unknown error'}`)
      }
    } catch (error: any) {
      console.error('Failed to process Google Sheets:', error)

      let errorMessage = 'Failed to process Google Sheets. Please check the URL and try again.'

      if (error.response?.status === 500) {
        errorMessage = 'Google Sheets processing is currently unavailable. Vector database is not configured.'
      } else if (error.response?.data?.detail) {
        errorMessage = `Processing failed: ${error.response.data.detail}`
      } else if (error.response?.data?.message) {
        errorMessage = `Processing failed: ${error.response.data.message}`
      }

      setSheetsProcessingStatus(`ล้มเหลว: ${errorMessage}`)
      alert(`❌ ${errorMessage}`)
    } finally {
      setIsProcessing(false)
      // Clear status after 3 seconds
      setTimeout(() => {
        setSheetsProcessingStatus('')
      }, 3000)
    }
  }

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    const formData = new FormData()
    const fileNames = []

    // Add files one by one (single file upload endpoint)
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const fileFormData = new FormData()
      fileFormData.append('file', file)
      fileFormData.append('user_id', '1')

      try {
        setIsLoading(true)
        console.log(`Uploading file: ${file.name}`)

        // Use upload-only endpoint
        const response = await api.post('/api/documents/upload', fileFormData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })

        console.log('Upload response:', response.data)

        if (response.data.success) {
          fileNames.push(file.name)
        } else {
          alert(`❌ Upload failed for ${file.name}: ${response.data.message}`)
        }
      } catch (error: any) {
        console.error(`Upload failed for ${file.name}:`, error)
        const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred'
        alert(`❌ Upload failed for ${file.name}: ${errorMessage}`)
      }
    }

    // Reload documents list after all uploads
    loadDocuments()

    // Show summary
    if (fileNames.length > 0) {
      alert(`✅ Successfully uploaded ${fileNames.length} files:\n${fileNames.join('\n')}\n\nClick 'Process Data' to add them to the vector database.`)
    }

    setIsLoading(false)
  }

  const handleProcessData = async () => {
    // Get unprocessed documents
    const unprocessedDocuments = documents.filter(doc => !doc.processed)

    if (unprocessedDocuments.length === 0) {
      alert('ℹ️ No unprocessed documents found. All documents are already processed.')
      return
    }

    const fileIds = unprocessedDocuments.map(doc => doc.id)
    const fileNames = unprocessedDocuments.map(doc => doc.filename)

    try {
      setIsProcessing(true)
      console.log(`Processing ${fileIds.length} documents: ${fileNames.join(', ')}`)

      const response = await api.post('/api/process', fileIds, {
        params: {
          clear_before_upload: false,
          user_id: '1'
        },
        timeout: 300000, // 5 minutes timeout for processing
      })

      console.log('Process response:', response.data)

      if (response.data.success) {
        alert(`✅ Successfully processed ${fileNames.length} documents:\n${fileNames.join('\n')}\n\n${response.data.message}`)
        loadDocuments() // Reload to update processed status
      } else {
        alert(`❌ Processing failed: ${response.data.message}`)
      }
    } catch (error: any) {
      console.error('Process failed:', error)
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred'
      alert(`❌ Processing failed: ${errorMessage}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleChat = async () => {
    if (!currentMessage.trim() || !selectedDocument) return

    try {
      setIsLoading(true)
      const response = await api.post('/api/chat', {
        message: currentMessage,
        document_id: selectedDocument.id,
        provider: selectedProvider,
        model: selectedModel
      })

      setChatMessages([...chatMessages, {
        id: Date.now().toString(),
        user_message: currentMessage,
        bot_response: response.data.response,
        timestamp: new Date().toISOString(),
        session_id: 'default'
      }])
      setCurrentMessage('')
    } catch (error) {
      console.error('Chat failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // General Chat Function
  const handleGeneralChatSend = async () => {
    if (!generalChatInput.trim()) return

    const userMessage = generalChatInput.trim()
    setGeneralChatInput('')
    setIsGeneralChatLoading(true)

    // Add user message to chat
    setGeneralChatMessages(prev => [...prev, {
      id: Date.now().toString(),
      sender: 'user' as const,
      message: userMessage,
      timestamp: new Date().toISOString()
    }])

    try {
      // Call general chat API (no document_id required)
      const response = await api.post('/api/chat/general', {
        message: userMessage,
        provider: selectedProvider,
        model: selectedModel
      })

      // Add AI response to chat
      setGeneralChatMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'bot' as const,
        message: response.data.response,
        timestamp: new Date().toISOString()
      }])
    } catch (error: any) {
      console.error('General chat failed:', error)
      // Add error message to chat
      setGeneralChatMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'bot' as const,
        message: `❌ Error: ${error.response?.data?.detail || error.message || 'Failed to get response'}`,
        timestamp: new Date().toISOString()
      }])
    } finally {
      setIsGeneralChatLoading(false)
    }
  }

  // Memory Management Functions
  const loadMemoryStats = async () => {
    try {
      const response = await api.get('/api/memory/stats')
      setMemoryStats(response.data)
    } catch (error) {
      console.error('Failed to load memory stats:', error)
    }
  }

  const addMemory = async () => {
    try {
      await api.post('/api/memory/add', {
        memory_type: newMemoryType,
        content: newMemoryContent
      })
      setNewMemoryContent('')
      loadMemoryStats()
    } catch (error) {
      console.error('Failed to add memory:', error)
    }
  }

  const searchMemories = async () => {
    try {
      const response = await api.post('/api/memory/search', {
        query: memorySearchQuery
      })
      setMemorySearchResults(response.data.results || [])
    } catch (error) {
      console.error('Failed to search memories:', error)
    }
  }

  const consolidateMemories = async () => {
    try {
      await api.post('/api/memory/consolidate')
      loadMemoryStats()
    } catch (error) {
      console.error('Failed to consolidate memories:', error)
    }
  }

  // Social Bots Functions
  const addBotConfig = async () => {
    try {
      await api.post('/api/social-bots/config', {
        platform: newBotPlatform,
        bot_token: newBotToken,
        webhook_url: newBotWebhook,
        channels: newBotChannels.split(',').map(c => c.trim())
      })
      setNewBotToken('')
      setNewBotWebhook('')
      setNewBotChannels('')
      loadBotConfigs()
    } catch (error) {
      console.error('Failed to add bot config:', error)
    }
  }

  const loadBotConfigs = async () => {
    try {
      const response = await api.get('/api/social-bots/configs')
      setBotConfigs(response.data.configs || {})
    } catch (error) {
      console.error('Failed to load bot configs:', error)
    }
  }

  const sendBotMessage = async () => {
    try {
      await api.post('/api/social-bots/send', {
        platform: selectedBotPlatform,
        channel: selectedBotChannel,
        message: botMessage
      })
      setBotMessage('')
    } catch (error) {
      console.error('Failed to send bot message:', error)
    }
  }

  const testBotConnection = async (platform: string) => {
    try {
      await api.post(`/api/social-bots/test/${platform}`)
    } catch (error) {
      console.error('Failed to test bot connection:', error)
    }
  }

  const deleteBotConfig = async (platform: string) => {
    try {
      await api.delete(`/api/social-bots/config/${platform}`)
      loadBotConfigs()
    } catch (error) {
      console.error('Failed to delete bot config:', error)
    }
  }

  // Feedback Functions
  const openFeedbackModal = (chatMessage: any) => {
    setSelectedChatForFeedback(chatMessage)
    setShowFeedbackModal(true)
    setFeedbackRating(0)
    setFeedbackType('good')
    setFeedbackText('')
  }

  const submitFeedback = async () => {
    if (!selectedChatForFeedback || feedbackRating === 0) {
      alert('กรุณาให้คะแนนดาว')
      return
    }

    try {
      const feedbackData = {
        chat_id: selectedChatForFeedback.id,
        question: selectedChatForFeedback.user_message,
        answer: selectedChatForFeedback.bot_response,
        rating: feedbackRating,
        feedback_type: feedbackType,
        feedback_text: feedbackText,
        ai_provider: selectedProvider,
        model: selectedModel,
        user_id: '1'
      }

      const response = await api.post('/api/feedback', feedbackData)

      if (response.data.success) {
        alert('✅ บันทึกความคิดเห็นสำเร็จ')
        setShowFeedbackModal(false)
        fetchFeedbackStats()
      }
    } catch (error: any) {
      alert(`บันทึกความคิดเห็นล้มเหลว: ${error.response?.data?.detail || error.message}`)
    }
  }

  const fetchFeedbackStats = async () => {
    try {
      const response = await api.get('/api/feedback/stats')
      setFeedbackStats(response.data)
    } catch (error) {
      console.error('Error fetching feedback stats:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <DocumentTextIcon className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-xl font-semibold text-gray-900">RAG PDF System</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowAIConfigModal(true)}
                className="flex items-center px-3 py-2 text-sm bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
              >
                <CogIcon className="h-4 w-4 mr-2" />
                AI Config
              </button>
              <button
                onClick={() => setShowSheetsModal(true)}
                className="flex items-center px-3 py-2 text-sm bg-green-600 text-white rounded-md hover:bg-green-700"
              >
                <DocumentTextIcon className="h-4 w-4 mr-2" />
                Google Sheets
              </button>
              <button
                onClick={() => setShowMemoryModal(true)}
                className="flex items-center px-3 py-2 text-sm bg-purple-600 text-white rounded-md hover:bg-purple-700"
              >
                <CpuChipIcon className="h-4 w-4 mr-2" />
                Memory
              </button>
              <button
                onClick={() => setShowSocialBotsModal(true)}
                className="flex items-center px-3 py-2 text-sm bg-teal-600 text-white rounded-md hover:bg-teal-700"
              >
                <ShareIcon className="h-4 w-4 mr-2" />
                Social Bots
              </button>
              {user ? (
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-700">Welcome, {user.username}</span>
                  <button
                    onClick={handleLogout}
                    className="flex items-center px-3 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700"
                  >
                    <ArrowRightStartOnRectangleIcon className="h-4 w-4 mr-2" />
                    Logout
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowLoginModal(true)}
                  className="px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Login
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {user && (
          /* General Chat Section */
          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <ChatBubbleLeftRightIcon className="h-5 w-5 mr-2 text-blue-600" />
              💬 AI Chat Assistant
            </h3>

            {/* Chat Messages */}
            <div className="border rounded-lg p-4 h-80 overflow-y-auto mb-4 bg-gray-50">
              {generalChatMessages.length === 0 ? (
                <div className="text-center text-gray-500">
                  <ChatBubbleLeftRightIcon className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                  <p>No messages yet. Start a conversation with the AI!</p>
                  <p className="text-sm mt-1">Ask questions about your uploaded documents or anything else.</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {generalChatMessages.map((msg, index) => (
                    <div
                      key={index}
                      className={`flex ${
                        msg.sender === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                          msg.sender === 'user'
                            ? 'bg-blue-500 text-white'
                            : 'bg-gray-200 text-gray-800'
                        }`}
                      >
                        <p className="text-sm whitespace-pre-wrap">{msg.message}</p>
                        <p className="text-xs mt-1 opacity-70">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))}
                  {isGeneralChatLoading && (
                    <div className="flex justify-start">
                      <div className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg flex items-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                        <p className="text-sm">Thinking...</p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Chat Input */}
            <div className="flex space-x-2">
              <input
                type="text"
                value={generalChatInput}
                onChange={(e) => setGeneralChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleGeneralChatSend()}
                placeholder="Ask anything about your documents or general questions..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isGeneralChatLoading}
              />
              <button
                onClick={handleGeneralChatSend}
                disabled={isGeneralChatLoading || !generalChatInput.trim()}
                className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center"
              >
                <PaperAirplaneIcon className="h-4 w-4 mr-2" />
                {isGeneralChatLoading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Documents Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-medium text-gray-900">Documents</h2>
                <div className="flex space-x-2">
                  <label className="flex items-center px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 cursor-pointer">
                    <CloudArrowUpIcon className="h-4 w-4 mr-2" />
                    Upload Documents
                    <input
                      type="file"
                      multiple
                      accept=".pdf,.txt,.docx,.xlsx,.xls,.jpg,.jpeg,.png,.gif,.bmp"
                      onChange={handleUpload}
                      className="hidden"
                    />
                  </label>
                  <button
                    onClick={handleProcessData}
                    disabled={isProcessing}
                    className="flex items-center px-3 py-2 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    <CpuChipIcon className="h-4 w-4 mr-2" />
                    {isProcessing ? 'Processing...' : 'Process Data'}
                  </button>
                </div>
              </div>
              <div className="space-y-2">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                      selectedDocument?.id === doc.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedDocument(doc)}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h3 className="text-sm font-medium text-gray-900 truncate">{doc.filename}</h3>
                        <p className="text-xs text-gray-500">{new Date(doc.upload_time).toLocaleDateString()}</p>
                        <div className="flex items-center space-x-2 mt-1">
                          {doc.processed ? (
                            <span className="inline-flex items-center px-2 py-1 text-xs font-medium text-green-700 bg-green-100 rounded-full">
                              ✅ Processed
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2 py-1 text-xs font-medium text-orange-700 bg-orange-100 rounded-full">
                              ⏳ Pending
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center space-x-1">
                        <button
                          className={`p-1 rounded ${
                            doc.is_favorite ? 'text-yellow-500' : 'text-gray-400'
                          } hover:text-yellow-500`}
                        >
                          <StarIcon className="h-4 w-4" />
                        </button>
                        {doc.file_type === 'application/pdf' && (
                          <DocumentTextIcon className="h-4 w-4 text-red-500" />
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Chat Panel */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow h-full flex flex-col">
              <div className="p-6 border-b">
                <h2 className="text-lg font-medium text-gray-900">
                  {selectedDocument ? `Chat with ${selectedDocument.filename}` : 'Select a document to start chatting'}
                </h2>
              </div>
              <div className="flex-1 p-6 overflow-y-auto">
                <div className="space-y-4">
                  {chatMessages.map((message) => (
                    <div key={message.id} className="space-y-2">
                      <div className="flex justify-end">
                        <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-xs">
                          {message.user_message}
                        </div>
                      </div>
                      <div className="flex justify-start">
                        <div className="bg-gray-100 text-gray-900 rounded-lg px-4 py-2 max-w-xs">
                          {message.bot_response}
                        </div>
                      </div>
                      <div className="flex justify-start">
                        <button
                          onClick={() => openFeedbackModal(message)}
                          className="text-gray-500 hover:text-blue-600 text-sm flex items-center space-x-1"
                        >
                          <StarIcon className="h-4 w-4" />
                          <span>Feedback</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {selectedDocument && (
                <div className="p-6 border-t">
                  <div className="flex space-x-4">
                    <input
                      type="text"
                      value={currentMessage}
                      onChange={(e) => setCurrentMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleChat()}
                      placeholder="Ask a question about the document..."
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={handleChat}
                      disabled={isLoading || !currentMessage.trim()}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                    >
                      {isLoading ? 'Sending...' : 'Send'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Memory Management Modal */}
      {showMemoryModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold flex items-center">
                  <CpuChipIcon className="h-6 w-6 mr-2 text-blue-600" />
                  Memory Management
                </h3>
                <button
                  onClick={() => setShowMemoryModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              <div className="text-center text-gray-500 py-8">
                Memory management features coming soon...
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Login Modal */}
      {showLoginModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-md w-full">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-gray-900">Login</h3>
                <button
                  onClick={() => {
                    setShowLoginModal(false)
                    setLoginError('')
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Username
                  </label>
                  <input
                    type="text"
                    value={loginUsername}
                    onChange={(e) => setLoginUsername(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter username"
                    onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Password
                  </label>
                  <input
                    type="password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter password"
                    onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
                  />
                </div>

                {loginError && (
                  <div className="text-red-600 text-sm">
                    {loginError}
                  </div>
                )}

                <div className="flex space-x-3 pt-4">
                  <button
                    onClick={handleLogin}
                    disabled={!loginUsername.trim() || !loginPassword.trim()}
                    className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Login
                  </button>
                  <button
                    onClick={() => {
                      setShowLoginModal(false)
                      setLoginError('')
                    }}
                    className="flex-1 px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300"
                  >
                    Cancel
                  </button>
                </div>

                <div className="text-xs text-gray-500 text-center pt-2">
                  Default credentials: admin / admin123
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Configuration Modal */}
      {showAIConfigModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold flex items-center">
                  <CogIcon className="h-6 w-6 mr-2 text-indigo-600" />
                  AI Configuration
                </h3>
                <button
                  onClick={() => setShowAIConfigModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {/* AI Provider Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    AI Provider
                  </label>
                  <select
                    value={selectedProvider}
                    onChange={(e) => handleProviderChange(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  >
                    {availableProviders.map((provider) => (
                      <option key={provider} value={provider}>
                        {provider.charAt(0).toUpperCase() + provider.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Model Selection */}
                {providerModels.length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Model
                    </label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    >
                      {providerModels.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <div className="bg-gray-50 p-4 rounded-md">
                  <h4 className="font-medium text-gray-900 mb-2">Current Configuration</h4>
                  <div className="text-sm text-gray-600">
                    <p>Provider: <span className="font-medium">{selectedProvider}</span></p>
                    <p>Model: <span className="font-medium">{selectedModel || 'Loading...'}</span></p>
                    <p>API Key: <span className="font-medium">{apiKeys[selectedProvider] ? '✅ Configured' : '❌ Not Set'}</span></p>
                  </div>
                </div>

                {/* API Key Management */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="block text-sm font-medium text-gray-700">
                      API Key Management
                    </label>
                    {!apiKeys[selectedProvider] && (
                      <button
                        onClick={() => setShowApiKeyInput(true)}
                        className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                      >
                        Add API Key
                      </button>
                    )}
                  </div>

                  {showApiKeyInput && !apiKeys[selectedProvider] && (
                    <div className="border border-gray-200 rounded-md p-3 space-y-2">
                      <input
                        type="password"
                        value={newApiKey}
                        onChange={(e) => setNewApiKey(e.target.value)}
                        placeholder={`Enter ${selectedProvider} API key`}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm"
                      />
                      <div className="flex space-x-2">
                        <button
                          onClick={saveApiKey}
                          disabled={!newApiKey.trim()}
                          className="flex-1 px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => {
                            setShowApiKeyInput(false)
                            setNewApiKey('')
                          }}
                          className="flex-1 px-3 py-1 bg-gray-300 text-gray-700 rounded text-sm hover:bg-gray-400"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {apiKeys[selectedProvider] && (
                    <div className="border border-gray-200 rounded-md p-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-green-600">✅ API Key configured</span>
                        <button
                          onClick={() => deleteApiKey(selectedProvider)}
                          className="text-xs px-2 py-1 bg-red-600 text-white rounded hover:bg-red-700"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => setShowAIConfigModal(false)}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                  >
                    Done
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Google Sheets Modal */}
      {showSheetsModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold flex items-center">
                  <DocumentTextIcon className="h-6 w-6 mr-2 text-green-600" />
                  Google Sheets Processing
                </h3>
                <button
                  onClick={() => {
                    setShowSheetsModal(false)
                    setSheetsUrl('')
                    setSheetsPreview(null)
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {/* Google Sheets URL Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Google Sheets URL
                  </label>
                  <input
                    type="url"
                    value={sheetsUrl}
                    onChange={(e) => setSheetsUrl(e.target.value)}
                    placeholder="https://docs.google.com/spreadsheets/d/..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  />
                </div>

                {/* Preview and Process Buttons */}
                <div className="flex space-x-3">
                  <button
                    onClick={previewGoogleSheets}
                    disabled={!sheetsUrl.trim() || isProcessing}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                  >
                    {isProcessing ? 'Loading...' : 'Preview'}
                  </button>
                  <button
                    onClick={processGoogleSheets}
                    disabled={!sheetsUrl.trim() || isProcessing}
                    className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
                  >
                    {isProcessing ? 'Processing...' : 'Process & Add to Documents'}
                  </button>
                </div>

                {/* Processing Status */}
                {sheetsProcessingStatus && (
                  <div className={`mt-3 p-3 rounded-lg text-sm font-medium ${
                    sheetsProcessingStatus.includes('สำเร็จ')
                      ? 'bg-green-100 text-green-800 border border-green-200'
                      : sheetsProcessingStatus.includes('ล้มเหลว')
                      ? 'bg-red-100 text-red-800 border border-red-200'
                      : 'bg-blue-100 text-blue-800 border border-blue-200'
                  }`}>
                    <div className="flex items-center space-x-2">
                      {sheetsProcessingStatus.includes('กำลัง') && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      )}
                      {sheetsProcessingStatus.includes('สำเร็จ') && (
                        <svg className="h-4 w-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                      {sheetsProcessingStatus.includes('ล้มเหลว') && (
                        <svg className="h-4 w-4 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      )}
                      <span>{sheetsProcessingStatus}</span>
                    </div>
                  </div>
                )}

                {/* Preview Results */}
                {sheetsPreview && (
                  <div className="bg-gray-50 p-4 rounded-md">
                    <h4 className="font-medium text-gray-900 mb-2">Preview</h4>
                    <div className="text-sm text-gray-600">
                      <p>Sheet Name: {sheetsPreview.sheet_name}</p>
                      <p>Rows: {sheetsPreview.row_count}</p>
                      <p>Columns: {sheetsPreview.column_count}</p>
                      {sheetsPreview.sample_data && (
                        <div className="mt-3">
                          <p className="font-medium">Sample Data:</p>
                          <pre className="bg-white p-2 rounded border text-xs overflow-auto">
                            {JSON.stringify(sheetsPreview.sample_data, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
                  <p className="text-sm text-yellow-800">
                    <strong>Note:</strong> Make sure your Google Sheet is publicly accessible or shared with the service account.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Social Media Bots Modal */}
      {showSocialBotsModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold flex items-center">
                  <ShareIcon className="h-6 w-6 mr-2 text-blue-600" />
                  Social Media Bots
                </h3>
                <button
                  onClick={() => setShowSocialBotsModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              <div className="text-center text-gray-500 py-8">
                Social media bot features coming soon...
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Feedback Modal */}
      {showFeedbackModal && selectedChatForFeedback && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">Rate this response</h3>
                <button
                  onClick={() => setShowFeedbackModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="p-6 space-y-4">
              {/* Question */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Your question:</label>
                <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-800">
                  {selectedChatForFeedback.user_message}
                </div>
              </div>

              {/* Answer */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">AI Response:</label>
                <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-800">
                  {selectedChatForFeedback.bot_response}
                </div>
              </div>

              {/* Star Rating */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Rating:</label>
                <div className="flex space-x-2">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <StarIcon
                      key={star}
                      className={`h-8 w-8 cursor-pointer transition-colors ${
                        star <= feedbackRating ? 'text-yellow-400 fill-current' : 'text-gray-300'
                      }`}
                      onClick={() => setFeedbackRating(star)}
                    />
                  ))}
                </div>
              </div>

              {/* Feedback Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Feedback Type:</label>
                <select
                  value={feedbackType}
                  onChange={(e) => setFeedbackType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="good">Good</option>
                  <option value="bad">Bad</option>
                  <option value="improvement">Needs Improvement</option>
                </select>
              </div>

              {/* Feedback Text */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Additional comments:</label>
                <textarea
                  value={feedbackText}
                  onChange={(e) => setFeedbackText(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Any additional feedback..."
                />
              </div>

              {/* Submit Button */}
              <div className="flex space-x-3 pt-4">
                <button
                  onClick={() => setShowFeedbackModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={submitFeedback}
                  disabled={feedbackRating === 0}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Submit Feedback
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}