// MongoDB initialization script for RAG PDF System

// Switch to the rag_pdf database
db = db.getSiblingDB('rag_pdf');

// Create application user with password
db.createUser({
  user: 'rag_user',
  pwd: 'rag_password',
  roles: [
    {
      role: 'readWrite',
      db: 'rag_pdf'
    }
  ]
});

// Create collections and indexes
db.createCollection('users');
db.createCollection('documents');
db.createCollection('api_keys');
db.createCollection('chat_history');
db.createCollection('feedback');
db.createCollection('memory');

// Create indexes for better performance
db.users.createIndex({ "username": 1 }, { unique: true });
db.documents.createIndex({ "user_id": 1 });
db.documents.createIndex({ "filename": 1 });
db.documents.createIndex({ "upload_date": -1 });
db.api_keys.createIndex({ "provider": 1 }, { unique: true });
db.chat_history.createIndex({ "user_id": 1, "timestamp": -1 });
db.feedback.createIndex({ "user_id": 1 });
db.memory.createIndex({ "user_id": 1, "memory_type": 1 });

// Insert default admin user
db.users.insertOne({
  username: 'admin',
  email: 'admin@ragpdf.com',
  full_name: 'Administrator',
  role: 'admin',
  is_active: true,
  created_at: new Date(),
  updated_at: new Date()
});

print('MongoDB initialization completed successfully!');
print('Database: rag_pdf');
print('User: rag_user with readWrite permissions');
print('Default admin user created: admin');