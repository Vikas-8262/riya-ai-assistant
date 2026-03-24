import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(RiyaApp());

class RiyaApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Riya AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: Color(0xFF89b4fa),
        scaffoldBackgroundColor: Color(0xFF1e1e2e),
      ),
      home: ChatScreen(),
    );
  }
}

class Message {
  final String text;
  final bool isUser;
  final DateTime time;
  Message(this.text, this.isUser) : time = DateTime.now();
}

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<Message>    _messages   = [];
  final TextEditingController _ctrl  = TextEditingController();
  final ScrollController  _scroll    = ScrollController();
  bool   _isTyping  = false;
  String _userName  = '';

  // Change this to your Render.com URL after deployment!
  final String API_URL = 'http://10.0.2.2:5000';

  @override
  void initState() {
    super.initState();
    _messages.add(Message(
        'Hello! I am Riya 🌟 Ask me anything!', false));
  }

  Future<void> sendMessage(String text) async {
    if (text.trim().isEmpty) return;
    _ctrl.clear();

    setState(() {
      _messages.add(Message(text, true));
      _isTyping = true;
    });
    _scrollDown();

    try {
      final res = await http.post(
        Uri.parse('$API_URL/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message':   text,
          'user_name': _userName.isEmpty ? null : _userName,
        }),
      ).timeout(Duration(seconds: 30));

      final data = jsonDecode(res.body);

      if (data['name'] != null) {
        setState(() => _userName = data['name']);
      }

      setState(() {
        _isTyping = false;
        _messages.add(Message(data['response'], false));
      });
    } catch (e) {
      setState(() {
        _isTyping = false;
        _messages.add(Message(
            'Sorry I had trouble connecting! 😔', false));
      });
    }
    _scrollDown();
  }

  void _scrollDown() {
    Future.delayed(Duration(milliseconds: 100), () {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Color(0xFF181825),
        elevation: 0,
        title: Row(children: [
          CircleAvatar(
            backgroundColor: Color(0xFFcba6f7),
            radius: 18,
            child: Text('R',
                style: TextStyle(
                    color: Color(0xFF1e1e2e),
                    fontWeight: FontWeight.bold)),
          ),
          SizedBox(width: 10),
          Column(crossAxisAlignment: CrossAxisAlignment.start,
              children: [
            Text('Riya AI',
                style: TextStyle(
                    fontSize: 16, fontWeight: FontWeight.bold)),
            Text('● Online',
                style: TextStyle(
                    fontSize: 11, color: Color(0xFFa6e3a1))),
          ]),
        ]),
      ),
      body: Column(children: [
        Expanded(
          child: ListView.builder(
            controller: _scroll,
            padding: EdgeInsets.all(16),
            itemCount: _messages.length + (_isTyping ? 1 : 0),
            itemBuilder: (ctx, i) {
              if (_isTyping && i == _messages.length) {
                return _buildTyping();
              }
              return _buildMessage(_messages[i]);
            },
          ),
        ),
        _buildInput(),
      ]),
    );
  }

  Widget _buildMessage(Message msg) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: msg.isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!msg.isUser) ...[
            CircleAvatar(
              backgroundColor: Color(0xFFcba6f7),
              radius: 14,
              child: Text('R',
                  style: TextStyle(
                      color: Color(0xFF1e1e2e),
                      fontSize: 12,
                      fontWeight: FontWeight.bold)),
            ),
            SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              padding: EdgeInsets.symmetric(
                  horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: msg.isUser
                    ? Color(0xFF45475a)
                    : Color(0xFF313244),
                borderRadius: BorderRadius.only(
                  topLeft:     Radius.circular(16),
                  topRight:    Radius.circular(16),
                  bottomLeft:  Radius.circular(
                      msg.isUser ? 16 : 4),
                  bottomRight: Radius.circular(
                      msg.isUser ? 4 : 16),
                ),
              ),
              child: Text(msg.text,
                  style: TextStyle(
                      color: Color(0xFFcdd6f4),
                      fontSize: 14)),
            ),
          ),
          if (msg.isUser) ...[
            SizedBox(width: 8),
            CircleAvatar(
              backgroundColor: Color(0xFF89b4fa),
              radius: 14,
              child: Text(
                  _userName.isNotEmpty
                      ? _userName[0].toUpperCase()
                      : 'V',
                  style: TextStyle(
                      color: Color(0xFF1e1e2e),
                      fontSize: 12,
                      fontWeight: FontWeight.bold)),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildTyping() {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(children: [
        CircleAvatar(
          backgroundColor: Color(0xFFcba6f7),
          radius: 14,
          child: Text('R',
              style: TextStyle(
                  color: Color(0xFF1e1e2e),
                  fontSize: 12,
                  fontWeight: FontWeight.bold)),
        ),
        SizedBox(width: 8),
        Container(
          padding: EdgeInsets.symmetric(
              horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: Color(0xFF313244),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(children: [
            _dot(0), SizedBox(width: 4),
            _dot(300), SizedBox(width: 4),
            _dot(600),
          ]),
        ),
      ]),
    );
  }

  Widget _dot(int delay) {
    return TweenAnimationBuilder(
      tween: Tween<double>(begin: 0, end: 1),
      duration: Duration(milliseconds: 600),
      builder: (ctx, val, _) => Container(
        width: 6, height: 6,
        decoration: BoxDecoration(
          color: Color(0xFFcba6f7),
          borderRadius: BorderRadius.circular(3),
        ),
      ),
    );
  }

  Widget _buildInput() {
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Color(0xFF181825),
        border: Border(
            top: BorderSide(color: Color(0xFF585b70))),
      ),
      child: Row(children: [
        Expanded(
          child: TextField(
            controller: _ctrl,
            style: TextStyle(color: Color(0xFFcdd6f4)),
            decoration: InputDecoration(
              hintText:      'Message Riya...',
              hintStyle:     TextStyle(color: Color(0xFF6c7086)),
              filled:        true,
              fillColor:     Color(0xFF313244),
              border:        OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide:   BorderSide.none,
              ),
              contentPadding: EdgeInsets.symmetric(
                  horizontal: 20, vertical: 12),
            ),
            onSubmitted: sendMessage,
          ),
        ),
        SizedBox(width: 10),
        GestureDetector(
          onTap: () => sendMessage(_ctrl.text),
          child: Container(
            padding: EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Color(0xFF89b4fa),
              borderRadius: BorderRadius.circular(24),
            ),
            child: Icon(Icons.send,
                color: Color(0xFF1e1e2e), size: 20),
          ),
        ),
      ]),
    );
  }
}