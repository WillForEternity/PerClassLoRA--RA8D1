#ifndef MCU_CONSTRAINTS_H
#define MCU_CONSTRAINTS_H

// RA8D1 Memory Specs
#define RA8D1_SRAM_BUDGET_KB 1024
#define RA8D1_FLASH_BUDGET_KB 2048

// Our application's self-imposed SRAM limit for this simulation
#define APP_SRAM_LIMIT (1024 * 1024) // 1 MB (1024 KB)


#endif // MCU_CONSTRAINTS_H
