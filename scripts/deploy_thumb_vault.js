const hre = require("hardhat");

async function main() {
  const THUMB = "0x8Ed90B81A84d84232408716e378013b0BCECE4fe";
  const Thumb3M = "0xA9677cd7F1cf0f1f1b9369ec93c746a54D12eF9C";
  const Thumb6M = "0x2F0C0c3d53f894A496696A990B80C176Ae43c0d1";
  const Thumb1Y = "0x9fAe46354861C1c011d77A301F2db29231078D3d";
  const Thumb2Y = "0xbB875bfF219eE0829A95aACd4cf235487Ff24591";
  const Thumb5Y = "0x4A0974C4C1Eb2A3b1440778641F74D2dC868499D";
  const Thumb10Y = "0x7A8f938601B3E906da387727D8B09253d49324C0";
  const Thumb30Y = "0x9B60DD13ADAf0233cfa382a0b7E0a81ee32F21Eb";

  const ThumbVault = await hre.ethers.getContractFactory("ThumbVault");
  const vault = await ThumbVault.deploy(
    THUMB,
    Thumb3M,
    Thumb6M,
    Thumb1Y,
    Thumb2Y,
    Thumb5Y,
    Thumb10Y,
    Thumb30Y
  );
  await vault.deployed();
  console.log("ThumbVault deployed to:", vault.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
}); 