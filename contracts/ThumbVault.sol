// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ThumbVault is ERC20, Ownable {
    IERC20 public immutable thumb;
    IERC20 public immutable thumb3M;
    IERC20 public immutable thumb6M;
    IERC20 public immutable thumb1Y;
    IERC20 public immutable thumb2Y;
    IERC20 public immutable thumb5Y;
    IERC20 public immutable thumb10Y;
    IERC20 public immutable thumb30Y;

    // Allocation in basis points (10000 = 100%)
    uint256[7] public allocations; // 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y

    event Deposit(address indexed user, uint256 amount, uint256 sharesMinted);
    event Withdraw(address indexed user, uint256 sharesBurned, uint256 amountReturned);
    event Rebalance(uint256[7] newAllocations);

    constructor(
        address _thumb,
        address _thumb3M,
        address _thumb6M,
        address _thumb1Y,
        address _thumb2Y,
        address _thumb5Y,
        address _thumb10Y,
        address _thumb30Y
    ) ERC20("Thumb Vault Share", "TVS") {
        thumb = IERC20(_thumb);
        thumb3M = IERC20(_thumb3M);
        thumb6M = IERC20(_thumb6M);
        thumb1Y = IERC20(_thumb1Y);
        thumb2Y = IERC20(_thumb2Y);
        thumb5Y = IERC20(_thumb5Y);
        thumb10Y = IERC20(_thumb10Y);
        thumb30Y = IERC20(_thumb30Y);
        allocations = [uint256(1428), 1428, 1428, 1428, 1428, 1428, 1432]; // default: equal split
    }

    function deposit(uint256 amount) external {
        require(amount > 0, "Amount must be > 0");
        uint256 totalAssets = totalAssets();
        uint256 shares = totalSupply() == 0 ? amount : (amount * totalSupply()) / totalAssets;
        require(thumb.transferFrom(msg.sender, address(this), amount), "Transfer failed");
        _mint(msg.sender, shares);
        emit Deposit(msg.sender, amount, shares);
    }

    function withdraw(uint256 shares) external {
        require(shares > 0, "Shares must be > 0");
        require(balanceOf(msg.sender) >= shares, "Not enough shares");
        uint256 totalAssets = totalAssets();
        uint256 amount = (shares * totalAssets) / totalSupply();
        _burn(msg.sender, shares);
        require(thumb.transfer(msg.sender, amount), "Transfer failed");
        emit Withdraw(msg.sender, shares, amount);
    }

    function rebalance(uint256[7] calldata newAllocations) external onlyOwner {
        uint256 total;
        for (uint256 i = 0; i < 7; i++) {
            total += newAllocations[i];
        }
        require(total == 10000, "Allocations must sum to 10000");
        allocations = newAllocations;
        emit Rebalance(newAllocations);
        // Here you would add logic to swap THUMB for the bond tokens according to new allocations
        // For now, this is a placeholder for integration with a DEX or other mechanism
    }

    function getUserBalance(address user) external view returns (uint256) {
        // User's share of total assets (in THUMB)
        if (totalSupply() == 0) return 0;
        return (balanceOf(user) * totalAssets()) / totalSupply();
    }

    function getVaultAllocations() external view returns (uint256[7] memory) {
        return allocations;
    }

    function totalAssets() public view returns (uint256) {
        // For now, just the THUMB held by the vault
        // In a real implementation, you would also include the value of the bond tokens
        return thumb.balanceOf(address(this));
    }
} 