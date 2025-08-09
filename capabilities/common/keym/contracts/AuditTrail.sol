// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AuditTrail
 * @dev Smart contract for immutable audit trail storage
 * @author Nyimbi Odero - Datacraft
 */

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

contract AuditTrail is Ownable, ReentrancyGuard {
    using ECDSA for bytes32;
    
    // Event structures
    struct AuditEvent {
        bytes32 eventId;
        uint256 timestamp;
        string eventType;
        string tenantId;
        string userId;
        string resourceId;
        string action;
        string result;
        bytes32 hash;
        bytes32 previousHash;
        bytes signature;
    }
    
    struct AuditBlock {
        bytes32 blockId;
        uint256 blockNumber;
        uint256 timestamp;
        bytes32 previousBlockHash;
        bytes32 merkleRoot;
        bytes32[] eventHashes;
        uint256 nonce;
        bytes32 hash;
        bytes signature;
        address validator;
    }
    
    // Storage
    mapping(bytes32 => AuditBlock) public blocks;
    mapping(bytes32 => AuditEvent) public events;
    mapping(bytes32 => bool) public eventExists;
    mapping(bytes32 => bool) public blockExists;
    
    bytes32[] public blockHashes;
    mapping(address => bool) public authorizedValidators;
    mapping(string => bool) public authorizedTenants;
    
    // Configuration
    uint256 public maxBlockSize = 100;
    uint256 public minBlockInterval = 300; // 5 minutes
    uint256 public lastBlockTimestamp;
    
    // Events
    event AuditEventStored(
        bytes32 indexed eventId,
        string indexed tenantId,
        string indexed eventType,
        uint256 timestamp
    );
    
    event AuditBlockCreated(
        bytes32 indexed blockId,
        uint256 indexed blockNumber,
        bytes32 merkleRoot,
        uint256 eventCount,
        address validator
    );
    
    event ValidatorAuthorized(address indexed validator);
    event ValidatorRevoked(address indexed validator);
    event TenantAuthorized(string indexed tenantId);
    event TenantRevoked(string indexed tenantId);
    
    // Modifiers
    modifier onlyAuthorizedValidator() {
        require(authorizedValidators[msg.sender], "Not an authorized validator");
        _;
    }
    
    modifier validTenant(string memory tenantId) {
        require(authorizedTenants[tenantId], "Tenant not authorized");
        _;
    }
    
    constructor() {
        // Initial configuration
        authorizedValidators[msg.sender] = true;
        lastBlockTimestamp = block.timestamp;
        
        // Create genesis block
        _createGenesisBlock();
    }
    
    /**
     * @dev Store audit event on blockchain
     */
    function storeAuditEvent(
        bytes32 eventId,
        string memory eventType,
        string memory tenantId,
        string memory userId,
        string memory resourceId,
        string memory action,
        string memory result,
        bytes32 hash,
        bytes32 previousHash,
        bytes memory signature
    ) external onlyAuthorizedValidator validTenant(tenantId) nonReentrant {
        require(!eventExists[eventId], "Event already exists");
        require(hash != bytes32(0), "Invalid hash");
        
        // Verify signature
        bytes32 messageHash = keccak256(abi.encodePacked(
            eventId, eventType, tenantId, userId, resourceId, action, result, hash, previousHash
        ));
        require(_verifySignature(messageHash, signature), "Invalid signature");
        
        // Create audit event
        AuditEvent memory auditEvent = AuditEvent({
            eventId: eventId,
            timestamp: block.timestamp,
            eventType: eventType,
            tenantId: tenantId,
            userId: userId,
            resourceId: resourceId,
            action: action,
            result: result,
            hash: hash,
            previousHash: previousHash,
            signature: signature
        });
        
        // Store event
        events[eventId] = auditEvent;
        eventExists[eventId] = true;
        
        emit AuditEventStored(eventId, tenantId, eventType, block.timestamp);
    }
    
    /**
     * @dev Create audit block with events
     */
    function createAuditBlock(
        bytes32 blockId,
        bytes32 previousBlockHash,
        bytes32 merkleRoot,
        bytes32[] memory eventHashes,
        uint256 nonce,
        bytes32 hash,
        bytes memory signature
    ) external onlyAuthorizedValidator nonReentrant {
        require(!blockExists[blockId], "Block already exists");
        require(eventHashes.length > 0, "No events in block");
        require(eventHashes.length <= maxBlockSize, "Block size exceeded");
        require(
            block.timestamp >= lastBlockTimestamp + minBlockInterval,
            "Block interval not met"
        );
        
        // Verify all events exist
        for (uint256 i = 0; i < eventHashes.length; i++) {
            require(eventExists[eventHashes[i]], "Event not found");
        }
        
        // Verify block hash
        bytes32 calculatedHash = keccak256(abi.encodePacked(
            blockId,
            blockHashes.length, // block number
            block.timestamp,
            previousBlockHash,
            merkleRoot,
            nonce
        ));
        require(hash == calculatedHash, "Invalid block hash");
        
        // Verify signature
        require(_verifySignature(hash, signature), "Invalid block signature");
        
        // Create audit block
        AuditBlock memory auditBlock = AuditBlock({
            blockId: blockId,
            blockNumber: blockHashes.length,
            timestamp: block.timestamp,
            previousBlockHash: previousBlockHash,
            merkleRoot: merkleRoot,
            eventHashes: eventHashes,
            nonce: nonce,
            hash: hash,
            signature: signature,
            validator: msg.sender
        });
        
        // Store block
        blocks[blockId] = auditBlock;
        blockExists[blockId] = true;
        blockHashes.push(blockId);
        lastBlockTimestamp = block.timestamp;
        
        emit AuditBlockCreated(
            blockId,
            auditBlock.blockNumber,
            merkleRoot,
            eventHashes.length,
            msg.sender
        );
    }
    
    /**
     * @dev Get audit event by ID
     */
    function getAuditEvent(bytes32 eventId) 
        external 
        view 
        returns (AuditEvent memory) 
    {
        require(eventExists[eventId], "Event not found");
        return events[eventId];
    }
    
    /**
     * @dev Get audit block by ID
     */
    function getAuditBlock(bytes32 blockId) 
        external 
        view 
        returns (AuditBlock memory) 
    {
        require(blockExists[blockId], "Block not found");
        return blocks[blockId];
    }
    
    /**
     * @dev Get latest block
     */
    function getLatestBlock() external view returns (AuditBlock memory) {
        require(blockHashes.length > 0, "No blocks exist");
        bytes32 latestBlockId = blockHashes[blockHashes.length - 1];
        return blocks[latestBlockId];
    }
    
    /**
     * @dev Get block count
     */
    function getBlockCount() external view returns (uint256) {
        return blockHashes.length;
    }
    
    /**
     * @dev Verify blockchain integrity
     */
    function verifyChainIntegrity() external view returns (bool) {
        if (blockHashes.length == 0) return true;
        
        for (uint256 i = 1; i < blockHashes.length; i++) {
            bytes32 currentBlockId = blockHashes[i];
            bytes32 previousBlockId = blockHashes[i - 1];
            
            AuditBlock memory currentBlock = blocks[currentBlockId];
            AuditBlock memory previousBlock = blocks[previousBlockId];
            
            // Check if current block's previous hash matches previous block's hash
            if (currentBlock.previousBlockHash != previousBlock.hash) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * @dev Verify Merkle proof for event
     */
    function verifyMerkleProof(
        bytes32 eventHash,
        bytes32 merkleRoot,
        bytes32[] memory proof,
        uint256 index
    ) external pure returns (bool) {
        return _verifyMerkleProof(eventHash, merkleRoot, proof, index);
    }
    
    /**
     * @dev Admin: Authorize validator
     */
    function authorizeValidator(address validator) external onlyOwner {
        authorizedValidators[validator] = true;
        emit ValidatorAuthorized(validator);
    }
    
    /**
     * @dev Admin: Revoke validator
     */
    function revokeValidator(address validator) external onlyOwner {
        require(validator != owner(), "Cannot revoke owner");
        authorizedValidators[validator] = false;
        emit ValidatorRevoked(validator);
    }
    
    /**
     * @dev Admin: Authorize tenant
     */
    function authorizeTenant(string memory tenantId) external onlyOwner {
        authorizedTenants[tenantId] = true;
        emit TenantAuthorized(tenantId);
    }
    
    /**
     * @dev Admin: Revoke tenant
     */
    function revokeTenant(string memory tenantId) external onlyOwner {
        authorizedTenants[tenantId] = false;
        emit TenantRevoked(tenantId);
    }
    
    /**
     * @dev Admin: Update configuration
     */
    function updateConfiguration(
        uint256 _maxBlockSize,
        uint256 _minBlockInterval
    ) external onlyOwner {
        require(_maxBlockSize > 0, "Invalid block size");
        require(_minBlockInterval > 0, "Invalid block interval");
        
        maxBlockSize = _maxBlockSize;
        minBlockInterval = _minBlockInterval;
    }
    
    /**
     * @dev Create genesis block
     */
    function _createGenesisBlock() internal {
        bytes32 genesisBlockId = keccak256(abi.encodePacked("genesis"));
        bytes32[] memory emptyEvents = new bytes32[](0);
        
        AuditBlock memory genesisBlock = AuditBlock({
            blockId: genesisBlockId,
            blockNumber: 0,
            timestamp: block.timestamp,
            previousBlockHash: bytes32(0),
            merkleRoot: bytes32(0),
            eventHashes: emptyEvents,
            nonce: 0,
            hash: keccak256(abi.encodePacked("genesis", block.timestamp)),
            signature: new bytes(0),
            validator: msg.sender
        });
        
        blocks[genesisBlockId] = genesisBlock;
        blockExists[genesisBlockId] = true;
        blockHashes.push(genesisBlockId);
        
        emit AuditBlockCreated(genesisBlockId, 0, bytes32(0), 0, msg.sender);
    }
    
    /**
     * @dev Verify digital signature
     */
    function _verifySignature(bytes32 hash, bytes memory signature) 
        internal 
        view 
        returns (bool) 
    {
        address signer = hash.recover(signature);
        return authorizedValidators[signer];
    }
    
    /**
     * @dev Verify Merkle proof
     */
    function _verifyMerkleProof(
        bytes32 leaf,
        bytes32 merkleRoot,
        bytes32[] memory proof,
        uint256 index
    ) internal pure returns (bool) {
        bytes32 computedHash = leaf;
        
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];
            
            if (index % 2 == 0) {
                // Hash(current computed hash + current element of the proof)
                computedHash = keccak256(abi.encodePacked(computedHash, proofElement));
            } else {
                // Hash(current element of the proof + current computed hash)
                computedHash = keccak256(abi.encodePacked(proofElement, computedHash));
            }
            
            index = index / 2;
        }
        
        // Check if the computed hash (root) is equal to the provided root
        return computedHash == merkleRoot;
    }
}