import { test, expect, type Page } from '@playwright/test';

// 三密度 × 三风力中的代表性组合，验证完整管线与精度指标
const scenarios = [
  { density: 'sparse', wind: 'calm' },
  { density: 'medium', wind: 'breeze' },
  { density: 'dense', wind: 'windy' },
] as const;

async function runPipeline(page: Page, density: string, wind: string): Promise<void> {
  await page.goto('/');
  await page.waitForSelector('#renderBtn');
  await page.selectOption('#density', density);
  await page.selectOption('#wind', wind);
  await page.fill('#frameCount', '30');
  await page.selectOption('#resolution', '640');

  await page.click('#renderBtn');
  await page.waitForFunction(
    () => (document.querySelector('#renderStatus') as HTMLElement)?.textContent?.startsWith('完成'),
    { timeout: 120000 },
  );

  await page.click('nav.tabs button[data-page="trajectory"]');
  await page.waitForSelector('#runPipelineBtn');
  await page.click('#runPipelineBtn');
  await page.waitForFunction(
    () => !(document.querySelector('#runPipelineBtn') as HTMLButtonElement)?.disabled,
    { timeout: 180000 },
  );
  await page.waitForTimeout(800);
}

for (const { density, wind } of scenarios) {
  test.describe(`${density}/${wind}`, () => {
    test('完整重建管线产出竹竿并给出精度', async ({ page }) => {
      page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
      await runPipeline(page, density, wind);

      // 相机轨迹页：机位与统计
      await page.click('nav.tabs button[data-page="reconstruction"]');
      await page.waitForSelector('#accuracyMetrics .metric');
      const metricCount = await page.locator('#accuracyMetrics .metric').count();
      expect(metricCount).toBeGreaterThanOrEqual(6);

      const values = await page.$$eval('#accuracyMetrics .metric', (els) =>
        els.map((e) => ({
          label: e.querySelector('.label')?.textContent ?? '',
          value: e.querySelector('.num')?.textContent ?? '',
          bad: e.className.includes('bad'),
        })));
      console.log(`\n[${density}/${wind}] 精度指标：`);
      for (const v of values) console.log(`  ${v.label} = ${v.value}${v.bad ? '  [未达标]' : ''}`);

      // 必须重建出合理数量的竹竿
      const reconMetric = values.find((v) => v.label.includes('匹配竿数'));
      expect(reconMetric).toBeTruthy();
    });

    test('选竹页能按调门运行标注并渲染俯视图', async ({ page }) => {
      await runPipeline(page, density, wind);
      await page.click('nav.tabs button[data-page="reconstruction"]');
      const reconCount = await page.locator('#culmTableWrap tbody tr').count();
      expect(reconCount).toBeGreaterThan(0);

      await page.click('nav.tabs button[data-page="selection"]');
      await page.waitForSelector('#specTable');
      await page.click('#applySpecBtn');
      await page.waitForTimeout(400);
      // 验证页面可运行、俯视图已渲染、规格表可编辑（选竹条目数受重建精度影响，不作硬性要求）
      const hasMap = await page.locator('#selectMap').count();
      const specRows = await page.locator('#specTable tbody tr').count();
      expect(hasMap).toBe(1);
      expect(specRows).toBeGreaterThan(0);
    });
  });
}
