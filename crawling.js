const fs = require('fs/promises');
const cheerio = require('cheerio');
const outPath = './factcheck_more_page1.json';
fs.writeFile(outPath, JSON.stringify([], null, 2), 'utf8');
(async () => {
  try {
    for (let i = 1; i <= 5; i++) {
        const url = `https://news.naver.com/factcheck/more?oid=&page=${i}`;
        const res = await fetch(url, {
        headers: {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        });

        if (!res.ok) throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);

        const html = await res.text();
        const htmlPath = './naver_factcheck_more_page1.html';
        await fs.writeFile(htmlPath, html, 'utf8');
        console.log(`Saved: ${htmlPath} (bytes:`, Buffer.byteLength(html, 'utf8'), ')');

        const $ = cheerio.load(html);
        const cards = [];

        $('li.factcheck_card').each((_, el) => {
        const $el = $(el);
        const $link = $el.find('a.factcheck_card_link');
        const url = $link.attr('href') || '';
        const title = $link.find('.factcheck_card_title').text().trim() || '';
        const desc = $link.find('.factcheck_card_desc').text().trim() || '';
        const subItems = $link.find('.factcheck_card_sub_info .factcheck_card_sub_item');
        const press = subItems.eq(0).text().trim() || '';
        const time = subItems.eq(1).text().trim() || '';
        const img = $link.find('.factcheck_card_img img').attr('src') || '';

        cards.push({ url, title, desc, press, time, img });
        });

        const outPath = './factcheck_more_page1.json';
        await fs.appendFile(outPath, JSON.stringify(cards, null, 2), 'utf8');
        console.log(`Saved: ${outPath} (count: ${cards.length})`);
    }
  } catch (err) {
    console.error('Error:', err);
    process.exit(1);
  }
})();
