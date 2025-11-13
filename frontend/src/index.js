import { Elysia } from "elysia";
import { html } from "@elysiajs/html";
import { cookie } from "@elysiajs/cookie";
import { jwt } from "@elysiajs/jwt";
import axios from "axios";

// API Configuration
const API_BASE_URL = "http://localhost:8000";

// Create Elysia app
const app = new Elysia({ prefix: "/" })
  .use(html())
  .use(cookie())
  .use(
    jwt({
      name: "jwt",
      secret: process.env.JWT_SECRET || "your-super-secret-jwt-key"
    })
  )
  .decorate({
    axios: axios.create({
      baseURL: API_BASE_URL,
      headers: {
        "Content-Type": "application/json"
      }
    })
  });

// HTML Components
const Layout = ({ title, children }) => html`
  <!DOCTYPE html>
  <html lang="th">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@2.0.3"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/remixicon@4.3.0/fonts/remixicon.css" rel="stylesheet">
    <style>
      .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
      }
      .upload-zone {
        border: 2px dashed #e5e7eb;
        border-radius: 0.5rem;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
      }
      .upload-zone:hover {
        border-color: #3b82f6;
        background-color: #f0f9ff;
      }
      .upload-zone.dragover {
        border-color: #3b82f6;
        background-color: #dbeafe;
      }
      .chat-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 0.5rem;
        max-width: 80%;
      }
      .user-message {
        background-color: #3b82f6;
        color: white;
        margin-left: auto;
      }
      .ai-message {
        background-color: #f3f4f6;
        color: #1f2937;
      }
      .file-item {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
      }
      .file-item:hover {
        border-color: #3b82f6;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
      }
      .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    </style>
  </head>
  <body class="bg-gray-50">
    ${children}
  </body>
  </html>
`;

const Navigation = ({ user }) => html`
  <nav class="bg-white shadow-sm border-b">
    <div class="container mx-auto px-4">
      <div class="flex justify-between items-center h-16">
        <div class="flex items-center">
          <h1 class="text-2xl font-bold text-blue-600">
            <i class="ri-file-text-line"></i> RAG PDF
          </h1>
        </div>
        <div class="flex items-center space-x-4">
          ${user ? html`
            <span class="text-sm text-gray-600">
              <i class="ri-user-line"></i> ${user.full_name || user.username}
            </span>
            <button
              hx-post="/logout"
              hx-target="body"
              class="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-md text-sm"
            >
              <i class="ri-logout-box-line"></i> ออกจากระบบ
            </button>
          ` : html`
            <a href="/login" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md text-sm">
              <i class="ri-login-box-line"></i> เข้าสู่ระบบ
            </a>
          `}
        </div>
      </div>
    </div>
  </nav>
`;

// Login Page
const LoginPage = () => Layout({
  title: "Login - RAG PDF",
  children: html`
    ${Navigation({ user: null })}
    <main class="container mx-auto px-4 py-8">
      <div class="max-w-md mx-auto">
        <div class="bg-white rounded-lg shadow-md p-8">
          <div class="text-center mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-2">เข้าสู่ระบบ</h2>
            <p class="text-gray-600">กรุณาเข้าสู่ระบบเพื่อใช้งาน RAG PDF</p>
          </div>

          <form
            hx-post="/api/auth/login"
            hx-target="#login-result"
            hx-swap="innerHTML"
            class="space-y-6"
          >
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                ชื่อผู้ใช้
              </label>
              <input
                type="text"
                name="username"
                required
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="กรอกชื่อผู้ใช้"
                value="admin"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                รหัสผ่าน
              </label>
              <input
                type="password"
                name="password"
                required
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="กรอกรหัสผ่าน"
                value="admin123"
              />
            </div>

            <button
              type="submit"
              class="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-md transition duration-200"
            >
              <i class="ri-login-box-line"></i> เข้าสู่ระบบ
            </button>
          </form>

          <div id="login-result"></div>

          <div class="mt-6 text-center text-sm text-gray-500">
            <p>Demo: admin / admin123</p>
          </div>
        </div>
      </div>
    </main>
  `
});

// Main Dashboard
const Dashboard = ({ user, documents }) => Layout({
  title: "Dashboard - RAG PDF",
  children: html`
    ${Navigation({ user })}
    <main class="container mx-auto px-4 py-8">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- Upload Section -->
        <div class="lg:col-span-1">
          <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-semibold mb-4">
              <i class="ri-upload-cloud-line"></i> อัปโหลดเอกสาร
            </h3>

            <div
              class="upload-zone"
              id="upload-zone"
              hx-post="/api/documents/upload"
              hx-encoding="multipart/form-data"
              hx-target="#upload-result"
              hx-swap="innerHTML"
              ondrop="handleDrop(event)"
              ondragover="handleDragOver(event)"
              ondragleave="handleDragLeave(event)"
            >
              <input
                type="file"
                name="file"
                accept=".pdf,.txt,.docx"
                class="hidden"
                id="file-input"
                onchange="handleFileSelect(event)"
              />
              <i class="ri-upload-cloud-2-line text-4xl text-gray-400 mb-2"></i>
              <p class="text-gray-600 mb-2">ลากไฟล์มาวางที่นี่หรือคลิกเพื่อเลือก</p>
              <p class="text-sm text-gray-500">รองรับ PDF, TXT, DOCX</p>
            </div>

            <div id="upload-result" class="mt-4"></div>
          </div>
        </div>

        <!-- Documents List -->
        <div class="lg:col-span-2">
          <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-semibold mb-4">
              <i class="ri-file-list-line"></i> เอกสารที่อัปโหลด
            </h3>

            ${documents.length > 0 ? html`
              <div class="space-y-2">
                ${documents.map(doc => html`
                  <div class="file-item">
                    <div class="flex justify-between items-start">
                      <div class="flex-1">
                        <h4 class="font-medium text-gray-900">${doc.filename}</h4>
                        <p class="text-sm text-gray-500">
                          ขนาด: ${(doc.size / 1024).toFixed(1)} KB |
                          อัปโหลด: ${new Date(doc.upload_date).toLocaleDateString('th-TH')}
                        </p>
                      </div>
                      <button
                        hx-delete="/api/documents/${doc.id}"
                        hx-target="body"
                        hx-swap="innerHTML"
                        class="text-red-500 hover:text-red-700 p-2"
                        title="ลบเอกสาร"
                      >
                        <i class="ri-delete-bin-line"></i>
                      </button>
                    </div>
                  </div>
                `).join('')}
              </div>
            ` : html`
              <div class="text-center py-8 text-gray-500">
                <i class="ri-file-text-line text-4xl mb-2"></i>
                <p>ยังไม่มีเอกสารที่อัปโหลด</p>
              </div>
            `}
          </div>
        </div>
      </div>

      <!-- Chat Section -->
      <div class="mt-8">
        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-lg font-semibold mb-4">
            <i class="ri-chat-3-line"></i> ถามคำถามเกี่ยวกับเอกสาร
          </h3>

          <div id="chat-messages" class="mb-4 max-h-96 overflow-y-auto space-y-4">
            <div class="text-center text-gray-500 py-8">
              <i class="ri-chat-smile-line text-4xl mb-2"></i>
              <p>พิมพ์คำถามเพื่อเริ่มสนทนา</p>
            </div>
          </div>

          <div class="flex space-x-2">
            <input
              type="text"
              id="question-input"
              placeholder="พิมพ์คำถามของคุณ..."
              class="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              onkeypress="if(event.key === 'Enter') sendQuestion()"
            />
            <button
              onclick="sendQuestion()"
              class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-md"
            >
              <i class="ri-send-plane-line"></i> ส่ง
            </button>
          </div>
        </div>
      </div>
    </main>

    <script>
      // File upload handlers
      function handleDragOver(e) {
        e.preventDefault();
        document.getElementById('upload-zone').classList.add('dragover');
      }

      function handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('upload-zone').classList.remove('dragover');
      }

      function handleDrop(e) {
        e.preventDefault();
        document.getElementById('upload-zone').classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
          document.getElementById('file-input').files = files;
          uploadFile();
        }
      }

      function handleFileSelect(e) {
        uploadFile();
      }

      function uploadFile() {
        const fileInput = document.getElementById('file-input');
        if (fileInput.files.length > 0) {
          const formData = new FormData();
          formData.append('file', fileInput.files[0]);

          fetch('/api/documents/upload', {
            method: 'POST',
            body: formData
          })
          .then(response => response.json())
          .then(data => {
            if (data.success) {
              location.reload(); // Refresh to show new document
            } else {
              alert('อัปโหลดล้มเหลว: ' + (data.detail || 'Unknown error'));
            }
          })
          .catch(error => {
            console.error('Upload error:', error);
            alert('อัปโหลดล้มเหลว: Network error');
          });
        }
      }

      // Chat functionality
      function sendQuestion() {
        const input = document.getElementById('question-input');
        const question = input.value.trim();

        if (!question) return;

        const messagesDiv = document.getElementById('chat-messages');

        // Add user message
        const userMessage = html\`
          <div class="chat-message user-message">
            <p class="font-medium">คุณ:</p>
            <p>\${question}</p>
          </div>
        \`;
        messagesDiv.innerHTML += userMessage;

        // Clear input
        input.value = '';

        // Add loading message
        const loadingMessage = html\`
          <div class="chat-message ai-message">
            <div class="loading"></div>
            <p class="font-medium">กำลังคิดคำตอบ...</p>
          </div>
        \`;
        messagesDiv.innerHTML += loadingMessage;

        // Scroll to bottom
        messagesDiv.scrollTop = messagesDiv.scrollHeight;

        // Send to API
        fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: question
          })
        })
        .then(response => response.json())
        .then(data => {
          // Remove loading message
          const messages = messagesDiv.children;
          messages[messages.length - 1].remove();

          // Add AI response
          const aiMessage = html\`
            <div class="chat-message ai-message">
              <p class="font-medium">AI:</p>
              <p>\${data.answer}</p>
              <div class="mt-2 text-xs text-gray-500">
                \${new Date(data.timestamp).toLocaleTimeString('th-TH')}
              </div>
            </div>
          \`;
          messagesDiv.innerHTML += aiMessage;

          // Scroll to bottom
          messagesDiv.scrollTop = messagesDiv.scrollHeight;
        })
        .catch(error => {
          console.error('Chat error:', error);

          // Remove loading message
          const messages = messagesDiv.children;
          messages[messages.length - 1].remove();

          // Add error message
          const errorMessage = html\`
            <div class="chat-message ai-message text-red-500">
              <p class="font-medium">Error:</p>
              <p>ไม่สามารถเชื่อมต่อกับเซิร์เวอร์</p>
            </div>
          \`;
          messagesDiv.innerHTML += errorMessage;

          // Scroll to bottom
          messagesDiv.scrollTop = messagesDiv.scrollHeight;
        });
      }
    </script>
  `
});

// Routes
app.get("/", () => {
  const user = null; // Check authentication
  if (user) {
    return Dashboard({ user, documents: [] });
  }
  return LoginPage();
});

app.get("/login", () => LoginPage());

app.post("/api/auth/login", async ({ body, set, jwt }) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body)
    });

    const data = await response.json();

    if (data.success) {
      const token = await jwt.sign(data.user);
      set.cookie['auth-token'] = {
        value: token,
        httpOnly: true,
        maxAge: 7 * 86400 * 1000, // 7 days
        path: '/'
      };

      return html`
        <script>
          window.location.href = '/';
        </script>
      `;
    } else {
      return html`
        <div class="text-red-500 text-center">
          <i class="ri-error-warning-line"></i>
          <p>ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง</p>
        </div>
      `;
    }
  } catch (error) {
    return html`
      <div class="text-red-500 text-center">
        <i class="ri-error-warning-line"></i>
        <p>เกิดข้อผิดพลาดในการเชื่อมต่อ</p>
      </div>
    `;
  }
});

app.post("/logout", ({ set, removeCookie }) => {
  removeCookie('auth-token');
  return html`
    <script>
      window.location.href = '/login';
    </script>
  `;
});

app.post("/api/documents/upload", async ({ body }) => {
  // This would normally use the file from the request
  // For now, return a mock response
  return {
    success: true,
    document_id: "demo-" + Math.random().toString(36).substr(2, 9),
    filename: "demo.pdf",
    size: 1024,
    message: "Document uploaded successfully"
  };
});

app.get("/api/documents", () => {
  // Mock documents data
  return {
    documents: [],
    total: 0
  };
});

app.post("/api/chat", async ({ body }) => {
  // Mock chat response
  return {
    answer: `นี่คือคำตอบสำหรับคำถาม: "${body.question}" (Demo response)`,
    sources: [
      {
        source: "demo",
        text: "This is a demo response from the RAG system"
      }
    ],
    timestamp: new Date().toISOString()
  };
});

app.listen(3000, () => {
  console.log("🦊 Elysia is running at http://localhost:3000");
  console.log("🚀 FastAPI Backend should be running at http://localhost:8000");
});