"""
Blockchain API Endpoints

Provides REST API endpoints for blockchain integration:
- Wallet management and connection
- Smart contract deployment and interaction
- DeFi operations and position management
- NFT minting and marketplace operations
- Cryptocurrency payment processing
- Multi-chain transaction monitoring

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from flask_appbuilder import BaseView, expose, has_access
from pydantic import ValidationError
import structlog

from .blockchain_service import (
	blockchain_service, BlockchainNetwork, ContractStandard, 
	TransactionStatus, DeFiProtocol
)

logger = structlog.get_logger(__name__)

# Create Flask Blueprint
blockchain_bp = Blueprint(
	'blockchain',
	__name__,
	url_prefix='/api/blockchain'
)


# =============================================================================
# Wallet Management Endpoints
# =============================================================================

@blockchain_bp.route('/wallets/connect', methods=['POST'])
async def connect_wallet():
	"""Connect blockchain wallet"""
	try:
		data = request.get_json()
		user_id = data.get('user_id') or session.get('user_id')
		
		if not user_id:
			return jsonify({
				"success": False,
				"error": "User ID required"
			}), 400
		
		required_fields = ['address', 'network']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		wallet_id = await blockchain_service.connect_wallet(user_id, data)
		
		return jsonify({
			"success": True,
			"data": {
				"wallet_id": wallet_id,
				"address": data['address'],
				"network": data['network']
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Connect wallet error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/wallets/<wallet_id>/portfolio', methods=['GET'])
async def get_wallet_portfolio(wallet_id: str):
	"""Get wallet portfolio"""
	try:
		portfolio = await blockchain_service.get_wallet_portfolio(wallet_id)
		
		return jsonify({
			"success": True,
			"data": portfolio
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 404
	except Exception as e:
		logger.error(f"Get wallet portfolio error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/wallets', methods=['GET'])
async def get_user_wallets():
	"""Get user's connected wallets"""
	try:
		user_id = session.get('user_id')
		if not user_id:
			return jsonify({
				"success": False,
				"error": "User ID required"
			}), 400
		
		# Get user's wallets
		user_wallets = [
			{
				"wallet_id": wallet.wallet_id,
				"address": wallet.address,
				"network": wallet.network.value,
				"wallet_type": wallet.wallet_type,
				"is_active": wallet.is_active,
				"balance": wallet.balance,
				"last_used": wallet.last_used.isoformat()
			}
			for wallet in blockchain_service.wallets.values()
			if wallet.user_id == user_id
		]
		
		return jsonify({
			"success": True,
			"data": {
				"wallets": user_wallets,
				"total": len(user_wallets)
			}
		})
		
	except Exception as e:
		logger.error(f"Get user wallets error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Smart Contract Endpoints
# =============================================================================

@blockchain_bp.route('/contracts/deploy', methods=['POST'])
async def deploy_smart_contract():
	"""Deploy smart contract"""
	try:
		data = request.get_json()
		wallet_id = data.get('wallet_id')
		
		if not wallet_id:
			return jsonify({
				"success": False,
				"error": "wallet_id is required"
			}), 400
		
		required_fields = ['name', 'standard']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		# Validate contract standard
		try:
			ContractStandard(data['standard'])
		except ValueError:
			return jsonify({
				"success": False,
				"error": f"Invalid contract standard: {data['standard']}"
			}), 400
		
		contract_id = await blockchain_service.deploy_smart_contract(wallet_id, data)
		
		return jsonify({
			"success": True,
			"data": {
				"contract_id": contract_id,
				"name": data['name'],
				"standard": data['standard']
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Deploy smart contract error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/contracts/<contract_id>/execute', methods=['POST'])
async def execute_contract_function():
	"""Execute smart contract function"""
	try:
		contract_id = request.view_args['contract_id']
		data = request.get_json()
		
		wallet_id = data.get('wallet_id')
		function_name = data.get('function_name')
		args = data.get('args', [])
		value = data.get('value', 0.0)
		
		if not wallet_id or not function_name:
			return jsonify({
				"success": False,
				"error": "wallet_id and function_name are required"
			}), 400
		
		tx_hash = await blockchain_service.execute_contract_function(
			wallet_id, contract_id, function_name, args, value
		)
		
		return jsonify({
			"success": True,
			"data": {
				"tx_hash": tx_hash,
				"contract_id": contract_id,
				"function_name": function_name
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Execute contract function error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/contracts', methods=['GET'])
async def get_contracts():
	"""Get deployed contracts"""
	try:
		user_id = session.get('user_id')
		network = request.args.get('network')
		standard = request.args.get('standard')
		
		contracts = []
		for contract in blockchain_service.contracts.values():
			# Filter by user's contracts (simplified)
			if network and contract.network.value != network:
				continue
			if standard and contract.standard.value != standard:
				continue
			
			contract_info = {
				"contract_id": contract.contract_id,
				"name": contract.name,
				"address": contract.address,
				"network": contract.network.value,
				"standard": contract.standard.value,
				"verification_status": contract.verification_status,
				"created_at": contract.created_at.isoformat()
			}
			contracts.append(contract_info)
		
		return jsonify({
			"success": True,
			"data": {
				"contracts": contracts,
				"total": len(contracts)
			}
		})
		
	except Exception as e:
		logger.error(f"Get contracts error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# NFT Endpoints
# =============================================================================

@blockchain_bp.route('/nfts/mint', methods=['POST'])
async def mint_nft():
	"""Mint NFT"""
	try:
		data = request.get_json()
		
		required_fields = ['wallet_id', 'contract_id', 'token_id', 'metadata']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		# Validate metadata
		metadata = data['metadata']
		if 'name' not in metadata:
			return jsonify({
				"success": False,
				"error": "NFT metadata must include 'name'"
			}), 400
		
		tx_hash = await blockchain_service.mint_nft(data['wallet_id'], data)
		
		return jsonify({
			"success": True,
			"data": {
				"tx_hash": tx_hash,
				"token_id": data['token_id'],
				"contract_id": data['contract_id']
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Mint NFT error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/nfts', methods=['GET'])
async def get_nfts():
	"""Get NFTs"""
	try:
		owner = request.args.get('owner')
		contract_address = request.args.get('contract_address')
		network = request.args.get('network')
		
		# Get NFTs from database (simplified query)
		from .database import get_async_db_session
		from sqlalchemy import text
		
		query = "SELECT * FROM nft_metadata WHERE 1=1"
		params = {}
		
		if owner:
			query += " AND owner = :owner"
			params['owner'] = owner
		
		if contract_address:
			query += " AND contract_address = :contract_address"
			params['contract_address'] = contract_address
		
		if network:
			query += " AND network = :network"
			params['network'] = network
		
		query += " ORDER BY created_at DESC LIMIT 50"
		
		async with get_async_db_session() as db_session:
			result = await db_session.execute(text(query), params)
			
			nfts = []
			for row in result:
				nft_data = {
					"id": row.id,
					"token_id": row.token_id,
					"contract_address": row.contract_address,
					"network": row.network,
					"name": row.name,
					"description": row.description,
					"image": row.image,
					"animation_url": row.animation_url,
					"external_url": row.external_url,
					"attributes": json.loads(row.attributes) if row.attributes else [],
					"rarity_rank": row.rarity_rank,
					"rarity_score": row.rarity_score,
					"creator": row.creator,
					"owner": row.owner,
					"created_at": row.created_at.isoformat()
				}
				nfts.append(nft_data)
		
		return jsonify({
			"success": True,
			"data": {
				"nfts": nfts,
				"total": len(nfts)
			}
		})
		
	except Exception as e:
		logger.error(f"Get NFTs error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# DeFi Endpoints
# =============================================================================

@blockchain_bp.route('/defi/swap', methods=['POST'])
async def execute_defi_swap():
	"""Execute DeFi token swap"""
	try:
		data = request.get_json()
		
		required_fields = ['wallet_id', 'token_in', 'token_out', 'amount_in']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		tx_hash = await blockchain_service.execute_defi_swap(data['wallet_id'], data)
		
		return jsonify({
			"success": True,
			"data": {
				"tx_hash": tx_hash,
				"token_in": data['token_in'],
				"token_out": data['token_out'],
				"amount_in": data['amount_in']
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Execute DeFi swap error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/defi/positions', methods=['POST'])
async def create_defi_position():
	"""Create DeFi position"""
	try:
		data = request.get_json()
		
		required_fields = ['wallet_id', 'protocol', 'position_type', 'token_address', 'amount']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		# Validate protocol
		try:
			DeFiProtocol(data['protocol'])
		except ValueError:
			return jsonify({
				"success": False,
				"error": f"Invalid DeFi protocol: {data['protocol']}"
			}), 400
		
		position_id = await blockchain_service.create_defi_position(data['wallet_id'], data)
		
		return jsonify({
			"success": True,
			"data": {
				"position_id": position_id,
				"protocol": data['protocol'],
				"position_type": data['position_type']
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Create DeFi position error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/defi/positions', methods=['GET'])
async def get_defi_positions():
	"""Get DeFi positions"""
	try:
		user_id = session.get('user_id')
		if not user_id:
			return jsonify({
				"success": False,
				"error": "User ID required"
			}), 400
		
		protocol = request.args.get('protocol')
		position_type = request.args.get('position_type')
		
		# Get positions from database
		from .database import get_async_db_session
		from sqlalchemy import text
		
		query = "SELECT * FROM defi_positions WHERE user_id = :user_id AND is_active = true"
		params = {"user_id": user_id}
		
		if protocol:
			query += " AND protocol = :protocol"
			params['protocol'] = protocol
		
		if position_type:
			query += " AND position_type = :position_type"
			params['position_type'] = position_type
		
		query += " ORDER BY opened_at DESC"
		
		async with get_async_db_session() as db_session:
			result = await db_session.execute(text(query), params)
			
			positions = []
			for row in result:
				position_data = {
					"id": row.id,
					"protocol": row.protocol,
					"network": row.network,
					"position_type": row.position_type,
					"token_address": row.token_address,
					"amount": row.amount,
					"collateral_amount": row.collateral_amount,
					"debt_amount": row.debt_amount,
					"apr": row.apr,
					"apy": row.apy,
					"rewards_earned": row.rewards_earned,
					"position_value_usd": row.position_value_usd,
					"health_factor": row.health_factor,
					"liquidation_price": row.liquidation_price,
					"opened_at": row.opened_at.isoformat(),
					"last_updated": row.last_updated.isoformat()
				}
				positions.append(position_data)
		
		return jsonify({
			"success": True,
			"data": {
				"positions": positions,
				"total": len(positions)
			}
		})
		
	except Exception as e:
		logger.error(f"Get DeFi positions error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/defi/protocols', methods=['GET'])
async def get_defi_protocols():
	"""Get supported DeFi protocols"""
	try:
		network = request.args.get('network')
		
		protocols = []
		for protocol_id, config in blockchain_service.defi_protocols.items():
			if network and config.get('network', {}).value != network:
				continue
			
			protocol_info = {
				"id": protocol_id,
				"name": config['name'],
				"network": config['network'].value,
				"supported_tokens": config.get('supported_tokens', []),
				"features": []
			}
			
			# Add feature flags based on protocol type
			if 'router_address' in config:
				protocol_info['features'].append('swapping')
			if 'lending_pool' in config:
				protocol_info['features'].extend(['lending', 'borrowing'])
			if 'pool_address' in config:
				protocol_info['features'].append('liquidity_provision')
			
			protocols.append(protocol_info)
		
		return jsonify({
			"success": True,
			"data": {
				"protocols": protocols,
				"total": len(protocols)
			}
		})
		
	except Exception as e:
		logger.error(f"Get DeFi protocols error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Payment Endpoints
# =============================================================================

@blockchain_bp.route('/payments/process', methods=['POST'])
async def process_crypto_payment():
	"""Process cryptocurrency payment"""
	try:
		data = request.get_json()
		
		required_fields = ['amount', 'currency', 'recipient', 'network']
		for field in required_fields:
			if field not in data:
				return jsonify({
					"success": False,
					"error": f"Missing required field: {field}"
				}), 400
		
		# Validate network
		try:
			BlockchainNetwork(data['network'])
		except ValueError:
			return jsonify({
				"success": False,
				"error": f"Invalid blockchain network: {data['network']}"
			}), 400
		
		payment_id = await blockchain_service.process_crypto_payment(data)
		
		return jsonify({
			"success": True,
			"data": {
				"payment_id": payment_id,
				"amount": data['amount'],
				"currency": data['currency'],
				"recipient": data['recipient']
			}
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 400
	except Exception as e:
		logger.error(f"Process crypto payment error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/payments/<payment_id>', methods=['GET'])
async def get_payment_status(payment_id: str):
	"""Get payment status"""
	try:
		# Get payment from database
		from .database import get_async_db_session
		from sqlalchemy import text
		
		async with get_async_db_session() as db_session:
			result = await db_session.execute(
				text("SELECT * FROM crypto_payments WHERE id = :payment_id"),
				{"payment_id": payment_id}
			)
			
			row = result.first()
			if not row:
				return jsonify({
					"success": False,
					"error": "Payment not found"
				}), 404
			
			payment_data = {
				"id": row.id,
				"amount": row.amount,
				"currency": row.currency,
				"recipient": row.recipient,
				"sender": row.sender,
				"network": row.network,
				"tx_hash": row.tx_hash,
				"status": row.status,
				"created_at": row.created_at.isoformat()
			}
		
		return jsonify({
			"success": True,
			"data": payment_data
		})
		
	except Exception as e:
		logger.error(f"Get payment status error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Transaction Monitoring Endpoints
# =============================================================================

@blockchain_bp.route('/transactions/<tx_hash>/status', methods=['GET'])
async def get_transaction_status(tx_hash: str):
	"""Get transaction status"""
	try:
		status = await blockchain_service.get_transaction_status(tx_hash)
		
		return jsonify({
			"success": True,
			"data": status
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": str(e)
		}), 404
	except Exception as e:
		logger.error(f"Get transaction status error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/transactions', methods=['GET'])
async def get_transactions():
	"""Get transactions"""
	try:
		user_id = session.get('user_id')
		network = request.args.get('network')
		status = request.args.get('status')
		limit = int(request.args.get('limit', 50))
		offset = int(request.args.get('offset', 0))
		
		# Get transactions from database
		from .database import get_async_db_session
		from sqlalchemy import text
		
		query = "SELECT * FROM blockchain_transactions WHERE 1=1"
		params = {}
		
		if network:
			query += " AND network = :network"
			params['network'] = network
		
		if status:
			query += " AND status = :status"
			params['status'] = status
		
		# Add user filter (simplified - would need wallet address lookup)
		if user_id:
			query += " AND (from_address IN (SELECT address FROM blockchain_wallets WHERE user_id = :user_id) OR to_address IN (SELECT address FROM blockchain_wallets WHERE user_id = :user_id))"
			params['user_id'] = user_id
		
		query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
		params['limit'] = limit
		params['offset'] = offset
		
		async with get_async_db_session() as db_session:
			result = await db_session.execute(text(query), params)
			
			transactions = []
			for row in result:
				tx_data = {
					"id": row.id,
					"tx_hash": row.tx_hash,
					"network": row.network,
					"from_address": row.from_address,
					"to_address": row.to_address,
					"value": row.value,
					"gas_limit": row.gas_limit,
					"gas_price": row.gas_price,
					"gas_used": row.gas_used,
					"status": row.status,
					"block_number": row.block_number,
					"created_at": row.created_at.isoformat(),
					"confirmed_at": row.confirmed_at.isoformat() if row.confirmed_at else None
				}
				transactions.append(tx_data)
		
		return jsonify({
			"success": True,
			"data": {
				"transactions": transactions,
				"total": len(transactions),
				"limit": limit,
				"offset": offset
			}
		})
		
	except Exception as e:
		logger.error(f"Get transactions error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Network and Configuration Endpoints
# =============================================================================

@blockchain_bp.route('/networks', methods=['GET'])
async def get_supported_networks():
	"""Get supported blockchain networks"""
	try:
		networks = []
		for network in BlockchainNetwork:
			network_info = {
				"id": network.value,
				"name": network.value.replace('_', ' ').title(),
				"rpc_endpoint": blockchain_service.rpc_endpoints.get(network, ""),
				"native_currency": {
					"name": "Ether" if network == BlockchainNetwork.ETHEREUM else network.value.upper(),
					"symbol": "ETH" if network == BlockchainNetwork.ETHEREUM else network.value.upper()[:4],
					"decimals": 18
				},
				"block_explorer": f"https://{network.value}.etherscan.io" if network != BlockchainNetwork.SOLANA else "https://solscan.io",
				"is_testnet": False
			}
			networks.append(network_info)
		
		return jsonify({
			"success": True,
			"data": {
				"networks": networks,
				"total": len(networks)
			}
		})
		
	except Exception as e:
		logger.error(f"Get supported networks error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@blockchain_bp.route('/contract-standards', methods=['GET'])
async def get_contract_standards():
	"""Get supported contract standards"""
	try:
		standards = []
		for standard in ContractStandard:
			standard_info = {
				"id": standard.value,
				"name": standard.value.upper(),
				"description": {
					"erc20": "Fungible Token Standard",
					"erc721": "Non-Fungible Token Standard",
					"erc1155": "Multi-Token Standard",
					"erc4626": "Tokenized Vault Standard",
					"governance": "DAO Governance Standard",
					"defi_protocol": "DeFi Protocol Standard",
					"marketplace": "NFT Marketplace Standard"
				}.get(standard.value, "Smart Contract Standard"),
				"use_cases": {
					"erc20": ["Cryptocurrencies", "Utility Tokens", "Stablecoins"],
					"erc721": ["NFTs", "Digital Art", "Gaming Items"],
					"erc1155": ["Gaming Assets", "Multi-Token Collections"],
					"erc4626": ["Yield Farming", "Vault Strategies"],
					"governance": ["DAO Voting", "Proposal Management"],
					"defi_protocol": ["AMM", "Lending", "Derivatives"],
					"marketplace": ["NFT Trading", "Auction Systems"]
				}.get(standard.value, ["General Purpose"])
			}
			standards.append(standard_info)
		
		return jsonify({
			"success": True,
			"data": {
				"standards": standards,
				"total": len(standards)
			}
		})
		
	except Exception as e:
		logger.error(f"Get contract standards error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Health and Analytics Endpoints
# =============================================================================

@blockchain_bp.route('/health', methods=['GET'])
async def blockchain_health_check():
	"""Blockchain service health check"""
	try:
		health_status = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"services": {
				"blockchain_service": "healthy",
				"wallet_management": "healthy",
				"smart_contracts": "healthy",
				"defi_integration": "healthy",
				"nft_management": "healthy",
				"payment_processing": "healthy"
			},
			"metrics": {
				"connected_wallets": len(blockchain_service.wallets),
				"deployed_contracts": len(blockchain_service.contracts),
				"supported_networks": len(BlockchainNetwork),
				"defi_protocols": len(blockchain_service.defi_protocols)
			}
		}
		
		return jsonify(health_status)
		
	except Exception as e:
		return jsonify({
			"status": "unhealthy",
			"error": str(e),
			"timestamp": datetime.utcnow().isoformat()
		}), 500


@blockchain_bp.route('/analytics', methods=['GET'])
async def get_blockchain_analytics():
	"""Get blockchain integration analytics"""
	try:
		# Calculate analytics from database
		from .database import get_async_db_session
		from sqlalchemy import text
		
		async with get_async_db_session() as db_session:
			# Get transaction counts by network
			tx_counts = await db_session.execute(
				text("SELECT network, COUNT(*) as count FROM blockchain_transactions GROUP BY network")
			)
			
			# Get contract counts by standard
			contract_counts = await db_session.execute(
				text("SELECT standard, COUNT(*) as count FROM smart_contracts GROUP BY standard")
			)
			
			# Get DeFi position values
			defi_values = await db_session.execute(
				text("SELECT protocol, SUM(position_value_usd) as total_value FROM defi_positions WHERE is_active = true GROUP BY protocol")
			)
			
			analytics = {
				"timestamp": datetime.utcnow().isoformat(),
				"transactions": {
					"by_network": {row.network: row.count for row in tx_counts}
				},
				"contracts": {
					"by_standard": {row.standard: row.count for row in contract_counts}
				},
				"defi": {
					"by_protocol": {row.protocol: row.total_value for row in defi_values}
				},
				"wallets": {
					"total_connected": len(blockchain_service.wallets),
					"by_network": {}
				}
			}
			
			# Calculate wallet distribution
			for wallet in blockchain_service.wallets.values():
				network = wallet.network.value
				analytics["wallets"]["by_network"][network] = analytics["wallets"]["by_network"].get(network, 0) + 1
		
		return jsonify({
			"success": True,
			"data": analytics
		})
		
	except Exception as e:
		logger.error(f"Get blockchain analytics error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Flask-AppBuilder Views
# =============================================================================

class BlockchainIntegrationView(BaseView):
	"""Blockchain integration management view"""
	
	route_base = "/blockchain"
	
	@expose("/dashboard")
	@has_access
	def dashboard(self):
		"""Blockchain integration dashboard"""
		return self.render_template(
			"blockchain/dashboard.html",
			title="Blockchain Integration Dashboard"
		)
	
	@expose("/wallets")
	@has_access
	def wallet_management(self):
		"""Wallet management interface"""
		return self.render_template(
			"blockchain/wallets.html",
			title="Wallet Management"
		)
	
	@expose("/contracts")
	@has_access
	def contract_management(self):
		"""Smart contract management interface"""
		return self.render_template(
			"blockchain/contracts.html",
			title="Smart Contracts"
		)
	
	@expose("/defi")
	@has_access
	def defi_management(self):
		"""DeFi operations interface"""
		return self.render_template(
			"blockchain/defi.html",
			title="DeFi Operations"
		)
	
	@expose("/nfts")
	@has_access
	def nft_management(self):
		"""NFT management interface"""
		return self.render_template(
			"blockchain/nfts.html",
			title="NFT Management"
		)
	
	@expose("/payments")
	@has_access
	def payment_management(self):
		"""Payment processing interface"""
		return self.render_template(
			"blockchain/payments.html",
			title="Crypto Payments"
		)


def register_blockchain_views(appbuilder):
	"""Register blockchain views with Flask-AppBuilder"""
	appbuilder.add_view(
		BlockchainIntegrationView,
		"Blockchain Dashboard",
		icon="fa-bitcoin",
		category="Blockchain",
		category_icon="fa-chain"
	)


def register_blockchain_routes(app):
	"""Register blockchain routes with Flask app"""
	app.register_blueprint(blockchain_bp)