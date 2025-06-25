const { ethers } = require("hardhat");
require("dotenv").config();

async function main() {
  // Get the signer
  const [signer] = await ethers.getSigners();
  console.log("Using account:", signer.address);

  // Load contract ABIs
  const thumbTokenABI = require("../contracts/ThumbToken.json").abi;
  const vaultABI = require("../contracts/Vault.json").abi;

  // Contract addresses
  const thumbTokenAddress = "0x8Ed90B81A84d84232408716e378013b0BCECE4fe";
  const vaultAddress = "0x112931aE1B5611c25f2e958D7845eA4d4394ED0D";

  // Create contract instances
  const thumbToken = new ethers.Contract(thumbTokenAddress, thumbTokenABI, signer);
  const vault = new ethers.Contract(vaultAddress, vaultABI, signer);

  // Get balances
  const thumbBalance = await thumbToken.balanceOf(signer.address);
  const vaultBalance = await vault.getUserBalance(signer.address);

  console.log("THUMB Balance:", ethers.utils.formatEther(thumbBalance));
  console.log("Vault Balance:", ethers.utils.formatEther(vaultBalance));

  // Example: Approve vault to spend THUMB tokens
  const amount = ethers.utils.parseEther("100");
  console.log("\nApproving vault to spend THUMB tokens...");
  const approveTx = await thumbToken.approve(vaultAddress, amount);
  await approveTx.wait();
  console.log("Approval successful");

  // Example: Deposit THUMB tokens
  console.log("\nDepositing THUMB tokens...");
  const depositTx = await vault.deposit(amount);
  await depositTx.wait();
  console.log("Deposit successful");

  // Example: Get bond allocations
  const allocations = await vault.getBondAllocations();
  console.log("\nCurrent bond allocations:", allocations.map(a => a.toString()));

  // Example: Rebalance with equal weights
  const weights = [1428, 1428, 1428, 1428, 1428, 1428, 1420]; // Equal weights in basis points
  console.log("\nRebalancing with equal weights...");
  const rebalanceTx = await vault.rebalance(weights);
  await rebalanceTx.wait();
  console.log("Rebalance successful");

  // Example: Withdraw THUMB tokens
  console.log("\nWithdrawing THUMB tokens...");
  const withdrawTx = await vault.withdraw(amount);
  await withdrawTx.wait();
  console.log("Withdrawal successful");

  // Get final balances
  const finalThumbBalance = await thumbToken.balanceOf(signer.address);
  const finalVaultBalance = await vault.getUserBalance(signer.address);

  console.log("\nFinal Balances:");
  console.log("THUMB Balance:", ethers.utils.formatEther(finalThumbBalance));
  console.log("Vault Balance:", ethers.utils.formatEther(finalVaultBalance));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 