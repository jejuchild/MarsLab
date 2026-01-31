import * as cheerio from 'cheerio';
import * as fs from 'fs';
import * as path from 'path';
import pLimit from 'p-limit';

// Types
interface ObservationRecord {
  obs_id: string;
  title: string;
  detail_page_url: string;
  jpeg_url: string;
  lbl_url: string;
  local_jpeg_path: string;
  local_lbl_path: string;
  // Spatial information extracted from LBL
  center_latitude?: number;
  center_longitude?: number;
  corner_coordinates?: [number, number][];
  map_scale?: number;
}

interface CrawlStats {
  total_discovered: number;
  downloaded: number;
  skipped: number;
  failed: number;
  failed_ids: string[];
}

// Configuration
const BASE_URL = 'https://www.uahirise.org';
const SEARCH_URL = `${BASE_URL}/results.php?keyword=arcadia&order=release_date&submit=Search`;
const OUTPUT_DIR = './arcadia_hirise';
const JPEG_DIR = path.join(OUTPUT_DIR, 'jpeg');
const LBL_DIR = path.join(OUTPUT_DIR, 'lbl');
const INDEX_FILE = path.join(OUTPUT_DIR, 'index.json');
const GEOJSON_FILE = path.join(OUTPUT_DIR, 'index.geojson');
const CONCURRENCY_LIMIT = 3;
const RETRY_COUNT = 3;
const RETRY_DELAY = 2000;

// Stats
const stats: CrawlStats = {
  total_discovered: 0,
  downloaded: 0,
  skipped: 0,
  failed: 0,
  failed_ids: [],
};

// Utility: delay
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Utility: fetch with retries
async function fetchWithRetry(url: string, retries = RETRY_COUNT): Promise<Response> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'MarsLab-HiRISE-Crawler/1.0 (Research purposes)',
        },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return response;
    } catch (error) {
      if (attempt === retries) throw error;
      console.log(`  Retry ${attempt}/${retries} for ${url}`);
      await delay(RETRY_DELAY * attempt);
    }
  }
  throw new Error('Failed after retries');
}

// Utility: download file
async function downloadFile(url: string, destPath: string): Promise<boolean> {
  try {
    const response = await fetchWithRetry(url);
    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(destPath, buffer);
    return true;
  } catch (error) {
    console.error(`  Failed to download ${url}: ${error}`);
    return false;
  }
}

// Parse LBL file for spatial information
function parseLblFile(lblPath: string): Partial<ObservationRecord> {
  const result: Partial<ObservationRecord> = {};

  if (!fs.existsSync(lblPath)) return result;

  try {
    const content = fs.readFileSync(lblPath, 'utf-8');

    // Extract bounding box from actual image extent
    const maxLatMatch = content.match(/MAXIMUM_LATITUDE\s*=\s*([-\d.]+)/);
    const minLatMatch = content.match(/MINIMUM_LATITUDE\s*=\s*([-\d.]+)/);
    const eastLonMatch = content.match(/EASTERNMOST_LONGITUDE\s*=\s*([-\d.]+)/);
    const westLonMatch = content.match(/WESTERNMOST_LONGITUDE\s*=\s*([-\d.]+)/);

    if (maxLatMatch && minLatMatch && eastLonMatch && westLonMatch) {
      const maxLat = parseFloat(maxLatMatch[1]);
      const minLat = parseFloat(minLatMatch[1]);
      const eastLon = parseFloat(eastLonMatch[1]);
      const westLon = parseFloat(westLonMatch[1]);

      // Calculate center from bounding box
      result.center_latitude = (maxLat + minLat) / 2;
      result.center_longitude = (eastLon + westLon) / 2;

      // Create corner coordinates for polygon (counter-clockwise from SW)
      // Format: [lon, lat] for GeoJSON
      result.corner_coordinates = [
        [westLon, minLat], // SW
        [eastLon, minLat], // SE
        [eastLon, maxLat], // NE
        [westLon, maxLat], // NW
      ];
    } else {
      // Fallback to CENTER_LATITUDE/CENTER_LONGITUDE if bounding box not available
      const latMatch = content.match(/CENTER_LATITUDE\s*=\s*([-\d.]+)/);
      const lonMatch = content.match(/CENTER_LONGITUDE\s*=\s*([-\d.]+)/);
      if (latMatch) result.center_latitude = parseFloat(latMatch[1]);
      if (lonMatch) result.center_longitude = parseFloat(lonMatch[1]);
    }

    // Extract MAP_SCALE
    const scaleMatch = content.match(/MAP_SCALE\s*=\s*([\d.]+)/);
    if (scaleMatch) {
      result.map_scale = parseFloat(scaleMatch[1]);
    }
  } catch (error) {
    console.error(`  Failed to parse LBL ${lblPath}: ${error}`);
  }

  return result;
}

// Get all observation links from a search results page
function parseSearchResultsPage(html: string, currentPage: number): { obsLinks: string[]; nextPageUrl: string | null; totalPages: number } {
  const $ = cheerio.load(html);
  const obsLinks: string[] = [];

  // Find all observation links - they use relative paths like href='ESP_013034_2185'
  $('a').each((_, el) => {
    const href = $(el).attr('href');
    if (href && /^(ESP|PSP)_\d+_\d+$/.test(href)) {
      const fullUrl = `${BASE_URL}/${href}`;
      if (!obsLinks.includes(fullUrl)) {
        obsLinks.push(fullUrl);
      }
    }
  });

  // Find total pages from pagination links
  let totalPages = currentPage;
  const pageMatches = html.match(/page=(\d+)/g);
  if (pageMatches) {
    for (const match of pageMatches) {
      const pageNum = parseInt(match.replace('page=', ''));
      if (pageNum > totalPages) {
        totalPages = pageNum;
      }
    }
  }

  // Determine next page URL
  let nextPageUrl: string | null = null;
  const nextPage = currentPage + 1;
  if (nextPage <= totalPages) {
    nextPageUrl = `${BASE_URL}/results.php?keyword=arcadia&order=release_date&submit=Search&page=${nextPage}`;
  }

  return { obsLinks, nextPageUrl, totalPages };
}

// Parse observation detail page
function parseDetailPage(html: string, url: string): Partial<ObservationRecord> | null {
  const $ = cheerio.load(html);

  // Extract observation ID from URL
  const obsIdMatch = url.match(/(ESP_\d+_\d+|PSP_\d+_\d+)/);
  if (!obsIdMatch) {
    console.error(`  Could not extract observation ID from ${url}`);
    return null;
  }
  const obs_id = obsIdMatch[1];

  // Extract title
  const title = $('span.observation-title-milo').text().trim() ||
                $('h1.observation-title').text().trim() ||
                $('title').text().trim();

  // Find map-projected JPEG link (B&W)
  let jpeg_url = '';
  $('a.caption, a').each((_, el) => {
    const text = $(el).text().trim().toLowerCase();
    const href = $(el).attr('href') || '';
    if (text === 'map projected' && href.includes('.jpg') && href.includes('_RED')) {
      jpeg_url = href;
    }
  });

  // Fallback: find any map-projected link with RED in URL
  if (!jpeg_url) {
    $('a[href*="_RED"][href*=".jpg"]').each((_, el) => {
      const text = $(el).text().trim().toLowerCase();
      if (text.includes('map') || text.includes('projected')) {
        jpeg_url = $(el).attr('href') || '';
      }
    });
  }

  // Fallback: broader search for abrowse.jpg
  if (!jpeg_url) {
    $('a[href*="abrowse.jpg"]').each((_, el) => {
      const href = $(el).attr('href') || '';
      if (href.includes('_RED')) {
        jpeg_url = href;
      }
    });
  }

  // Find B&W label link
  let lbl_url = '';
  $('a.caption, a').each((_, el) => {
    const text = $(el).text().trim().toLowerCase();
    const href = $(el).attr('href') || '';
    if ((text === 'b&w label' || text === 'bw label' || text.includes('b&w label')) && href.includes('.LBL')) {
      lbl_url = href;
    }
  });

  // Fallback: find any RED.LBL link
  if (!lbl_url) {
    $('a[href*="_RED.LBL"]').each((_, el) => {
      lbl_url = $(el).attr('href') || '';
    });
  }

  if (!jpeg_url || !lbl_url) {
    console.log(`  Warning: Missing links for ${obs_id} - JPEG: ${!!jpeg_url}, LBL: ${!!lbl_url}`);
  }

  return {
    obs_id,
    title,
    detail_page_url: url,
    jpeg_url,
    lbl_url,
    local_jpeg_path: jpeg_url ? `jpeg/${obs_id}_RED.abrowse.jpg` : '',
    local_lbl_path: lbl_url ? `lbl/${obs_id}_RED.LBL` : '',
  };
}

// Process a single observation
async function processObservation(detailUrl: string, existingIds: Set<string>): Promise<ObservationRecord | null> {
  try {
    // Extract obs_id to check for duplicates early
    const obsIdMatch = detailUrl.match(/(ESP_\d+_\d+|PSP_\d+_\d+)/);
    if (obsIdMatch && existingIds.has(obsIdMatch[1])) {
      console.log(`  Skipping duplicate: ${obsIdMatch[1]}`);
      stats.skipped++;
      return null;
    }

    console.log(`  Fetching detail page: ${detailUrl}`);
    const response = await fetchWithRetry(detailUrl);
    const html = await response.text();

    const record = parseDetailPage(html, detailUrl);
    if (!record || !record.obs_id) {
      console.error(`  Failed to parse detail page: ${detailUrl}`);
      stats.failed++;
      return null;
    }

    if (existingIds.has(record.obs_id)) {
      console.log(`  Skipping duplicate: ${record.obs_id}`);
      stats.skipped++;
      return null;
    }

    existingIds.add(record.obs_id);
    console.log(`  Processing: ${record.obs_id} - ${record.title}`);

    // Download files
    let success = true;

    if (record.jpeg_url) {
      const jpegPath = path.join(OUTPUT_DIR, record.local_jpeg_path!);
      if (!fs.existsSync(jpegPath)) {
        console.log(`    Downloading JPEG...`);
        if (!await downloadFile(record.jpeg_url, jpegPath)) {
          success = false;
        }
      } else {
        console.log(`    JPEG already exists`);
      }
    } else {
      success = false;
    }

    if (record.lbl_url) {
      const lblPath = path.join(OUTPUT_DIR, record.local_lbl_path!);
      if (!fs.existsSync(lblPath)) {
        console.log(`    Downloading LBL...`);
        if (!await downloadFile(record.lbl_url, lblPath)) {
          success = false;
        }
      } else {
        console.log(`    LBL already exists`);
      }

      // Parse LBL for spatial info
      const lblPath2 = path.join(OUTPUT_DIR, record.local_lbl_path!);
      if (fs.existsSync(lblPath2)) {
        const spatialInfo = parseLblFile(lblPath2);
        Object.assign(record, spatialInfo);
      }
    } else {
      success = false;
    }

    if (success) {
      stats.downloaded++;
    } else {
      stats.failed++;
      stats.failed_ids.push(record.obs_id);
    }

    return record as ObservationRecord;
  } catch (error) {
    console.error(`  Error processing ${detailUrl}: ${error}`);
    stats.failed++;
    return null;
  }
}

// Generate GeoJSON from records
function generateGeoJSON(records: ObservationRecord[]): object {
  const features = records
    .filter(r => r.center_latitude !== undefined && r.center_longitude !== undefined)
    .map(record => {
      // Create polygon if corner coordinates available, otherwise point
      let geometry: any;

      if (record.corner_coordinates && record.corner_coordinates.length === 4) {
        // Close the polygon by repeating the first point
        const coords = [...record.corner_coordinates, record.corner_coordinates[0]];
        geometry = {
          type: 'Polygon',
          coordinates: [coords],
        };
      } else {
        geometry = {
          type: 'Point',
          coordinates: [record.center_longitude, record.center_latitude],
        };
      }

      return {
        type: 'Feature',
        geometry,
        properties: {
          obs_id: record.obs_id,
          title: record.title,
          detail_page_url: record.detail_page_url,
          jpeg_url: record.jpeg_url,
          lbl_url: record.lbl_url,
          local_jpeg_path: record.local_jpeg_path,
          local_lbl_path: record.local_lbl_path,
          center_latitude: record.center_latitude,
          center_longitude: record.center_longitude,
          map_scale: record.map_scale,
        },
      };
    });

  return {
    type: 'FeatureCollection',
    features,
  };
}

// Main crawler function
async function crawl(): Promise<void> {
  console.log('=== HiRISE Arcadia Crawler ===\n');

  // Create output directories
  fs.mkdirSync(JPEG_DIR, { recursive: true });
  fs.mkdirSync(LBL_DIR, { recursive: true });

  // Load existing records if any
  const existingRecords: ObservationRecord[] = fs.existsSync(INDEX_FILE)
    ? JSON.parse(fs.readFileSync(INDEX_FILE, 'utf-8'))
    : [];
  const existingIds = new Set(existingRecords.map(r => r.obs_id));
  console.log(`Loaded ${existingRecords.length} existing records\n`);

  const allRecords: ObservationRecord[] = [...existingRecords];
  const limit = pLimit(CONCURRENCY_LIMIT);

  // Crawl search results pages
  let currentUrl: string | null = SEARCH_URL;
  let pageNum = 1;
  let totalPages = 1;
  const allDetailUrls: string[] = [];

  while (currentUrl) {
    console.log(`\n--- Page ${pageNum}${totalPages > 1 ? ` of ${totalPages}` : ''} ---`);
    console.log(`Fetching: ${currentUrl}`);

    try {
      const response = await fetchWithRetry(currentUrl);
      const html = await response.text();
      const result = parseSearchResultsPage(html, pageNum);

      console.log(`Found ${result.obsLinks.length} observation links`);
      allDetailUrls.push(...result.obsLinks);
      totalPages = result.totalPages;
      currentUrl = result.nextPageUrl;
      pageNum++;

      // Be polite
      await delay(1000);
    } catch (error) {
      console.error(`Failed to fetch page: ${error}`);
      break;
    }
  }

  // Deduplicate URLs
  const uniqueUrls = [...new Set(allDetailUrls)];
  stats.total_discovered = uniqueUrls.length;
  console.log(`\n=== Processing ${uniqueUrls.length} unique observations ===\n`);

  // Process observations with concurrency limit
  const tasks = uniqueUrls.map(url =>
    limit(async () => {
      const record = await processObservation(url, existingIds);
      if (record) {
        allRecords.push(record);
      }
      // Be polite between requests
      await delay(500);
    })
  );

  await Promise.all(tasks);

  // Remove duplicates and save
  const uniqueRecords = Array.from(
    new Map(allRecords.map(r => [r.obs_id, r])).values()
  );

  // Save index.json
  fs.writeFileSync(INDEX_FILE, JSON.stringify(uniqueRecords, null, 2));
  console.log(`\nSaved ${uniqueRecords.length} records to ${INDEX_FILE}`);

  // Generate and save GeoJSON
  const geojson = generateGeoJSON(uniqueRecords);
  fs.writeFileSync(GEOJSON_FILE, JSON.stringify(geojson, null, 2));
  console.log(`Saved GeoJSON to ${GEOJSON_FILE}`);

  // Print stats
  console.log('\n=== Crawl Statistics ===');
  console.log(`Total discovered: ${stats.total_discovered}`);
  console.log(`Successfully downloaded: ${stats.downloaded}`);
  console.log(`Skipped (duplicates): ${stats.skipped}`);
  console.log(`Failed: ${stats.failed}`);
  if (stats.failed_ids.length > 0) {
    console.log(`Failed IDs: ${stats.failed_ids.join(', ')}`);
  }
}

// Run
crawl().catch(console.error);
