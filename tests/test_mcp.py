#!/usr/bin/env python3
"""
Debug Test Client for MCP Server

Test script to verify MCP server is working correctly and debug any issues
"""

import asyncio
import json
import logging
import requests
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPDebugClient:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip("/")
        
    def test_health(self):
        """Test the health endpoint"""
        try:
            logger.info("🔍 Testing health endpoint...")
            response = requests.get(f"{self.base_url}/health")
            logger.info(f"Health status: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Health response: {response.json()}")
                return True
            else:
                logger.error(f"Health check failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
            
    def test_server_info(self):
        """Test the server info endpoint"""
        try:
            logger.info("🔍 Testing server info endpoint...")
            response = requests.get(f"{self.base_url}/mcp/info")
            logger.info(f"Server info status: {response.status_code}")
            if response.status_code == 200:
                info = response.json()
                logger.info(f"Server info: {json.dumps(info, indent=2)}")
                logger.info(f"Available tools: {[tool['name'] for tool in info.get('tools', [])]}")
                return True
            else:
                logger.error(f"Server info failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Server info error: {e}")
            return False
            
    def test_tool_call_direct(self, tool_name, arguments):
        """Test direct tool call endpoint"""
        try:
            logger.info(f"🔧 Testing direct tool call: {tool_name}")
            logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")
            
            response = requests.post(
                f"{self.base_url}/mcp/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments
                },
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"Tool call status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Tool call result: {json.dumps(result, indent=2)}")
                return True, result
            else:
                logger.error(f"Tool call failed: {response.text}")
                return False, None
                
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return False, None
            
    def test_mcp_rpc(self, method, params):
        """Test MCP RPC endpoint"""
        try:
            logger.info(f"🔌 Testing MCP RPC: {method}")
            logger.info(f"Params: {json.dumps(params, indent=2)}")
            
            rpc_request = {
                "jsonrpc": "2.0",
                "id": f"test_{datetime.utcnow().timestamp()}",
                "method": method,
                "params": params
            }
            
            response = requests.post(
                f"{self.base_url}/mcp/rpc",
                json=rpc_request,
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"MCP RPC status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"MCP RPC result: {json.dumps(result, indent=2)}")
                return True, result
            else:
                logger.error(f"MCP RPC failed: {response.text}")
                return False, None
                
        except Exception as e:
            logger.error(f"MCP RPC error: {e}")
            return False, None

def main():
    """Run debug tests"""
    logger.info("🚀 Starting MCP Debug Tests")
    
    client = MCPDebugClient()
    
    # Test 1: Health check
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Health Check")
    logger.info("="*50)
    if not client.test_health():
        logger.error("❌ Health check failed - server may not be running")
        return
    
    # Test 2: Server info
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Server Info")
    logger.info("="*50)
    if not client.test_server_info():
        logger.error("❌ Server info failed")
        return
    
    # Test 3: Simple document search (direct tool call)
    logger.info("\n" + "="*50)
    logger.info("TEST 3: Document Search (Direct Tool Call)")
    logger.info("="*50)
    success, result = client.test_tool_call_direct(
        "search_financial_documents",
        {
            "query": "revenue growth",
            "top_k": 5
        }
    )
    
    if not success:
        logger.error("❌ Document search tool call failed")
    else:
        logger.info("✅ Document search tool call succeeded")
    
    # Test 4: Financial question (direct tool call)
    logger.info("\n" + "="*50)
    logger.info("TEST 4: Financial Question (Direct Tool Call)")
    logger.info("="*50)
    success, result = client.test_tool_call_direct(
        "answer_financial_question",
        {
            "question": "What are the key financial risks mentioned in Apple's latest 10-K filing?",
            "verification_level": "thorough",
            "use_multi_agent": True
        }
    )
    
    if not success:
        logger.error("❌ Financial question tool call failed")
    else:
        logger.info("✅ Financial question tool call succeeded")
        if result and 'result' in result:
            result_data = result['result']
            logger.info(f"📊 Search results count: {result_data.get('search_results_count', 0)}")
            logger.info(f"📚 Sources count: {len(result_data.get('sources', []))}")
            logger.info(f"📝 Answer length: {len(result_data.get('answer', ''))}")
            logger.info(f"🎯 Method used: {result_data.get('method', 'unknown')}")
    
    # Test 5: Compare with a simpler question
    logger.info("\n" + "="*50)
    logger.info("TEST 5: Simple Financial Question (Direct Tool Call)")
    logger.info("="*50)
    success, result = client.test_tool_call_direct(
        "answer_financial_question",
        {
            "question": "What is the current revenue trend for tech companies?",
            "verification_level": "basic",
            "use_multi_agent": True
        }
    )
    
    if not success:
        logger.error("❌ Simple financial question tool call failed")
    else:
        logger.info("✅ Simple financial question tool call succeeded")
        if result and 'result' in result:
            result_data = result['result']
            logger.info(f"📊 Search results count: {result_data.get('search_results_count', 0)}")
            logger.info(f"📚 Sources count: {len(result_data.get('sources', []))}")
            logger.info(f"📝 Answer length: {len(result_data.get('answer', ''))}")
            logger.info(f"🎯 Method used: {result_data.get('method', 'unknown')}")
    
    # Test 6: MCP RPC call
    logger.info("\n" + "="*50)
    logger.info("TEST 6: MCP RPC Call")
    logger.info("="*50)
    success, result = client.test_mcp_rpc(
        "answer_financial_question",
        {
            "question": "What are the main financial risks mentioned in recent SEC filings?",
            "verification_level": "thorough",
            "use_multi_agent": True
        }
    )
    
    if not success:
        logger.error("❌ MCP RPC call failed")
    else:
        logger.info("✅ MCP RPC call succeeded")
    
    logger.info("\n🏁 Debug tests completed!")

if __name__ == "__main__":
    main()
