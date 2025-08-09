/**
 * Webpack Configuration for APG Workflow Orchestration UI
 * 
 * Build configuration for React drag-drop canvas interface.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  const isDevelopment = !isProduction;

  return {
    entry: {
      main: './js/index.js',
      workflow_canvas: './js/workflow_canvas.js'
    },

    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction ? 'js/[name].[contenthash].js' : 'js/[name].js',
      chunkFilename: isProduction ? 'js/[name].[contenthash].chunk.js' : 'js/[name].chunk.js',
      clean: true,
      publicPath: '/static/workflow_orchestration/'
    },

    resolve: {
      extensions: ['.js', '.jsx', '.json'],
      alias: {
        '@': path.resolve(__dirname, 'js'),
        '@components': path.resolve(__dirname, 'js/components'),
        '@utils': path.resolve(__dirname, 'js/utils'),
        '@styles': path.resolve(__dirname, 'css')
      }
    },

    module: {
      rules: [
        // JavaScript and JSX
        {
          test: /\.(js|jsx)$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: [
                ['@babel/preset-env', {
                  targets: {
                    browsers: ['> 1%', 'last 2 versions']
                  },
                  useBuiltIns: 'usage',
                  corejs: 3
                }],
                ['@babel/preset-react', {
                  runtime: 'automatic'
                }]
              ],
              plugins: [
                '@babel/plugin-proposal-class-properties',
                ['@babel/plugin-transform-runtime', {
                  regenerator: true
                }]
              ]
            }
          }
        },

        // CSS and SCSS
        {
          test: /\.css$/,
          use: [
            isDevelopment ? 'style-loader' : MiniCssExtractPlugin.loader,
            {
              loader: 'css-loader',
              options: {
                importLoaders: 1,
                modules: {
                  auto: true,
                  localIdentName: isDevelopment 
                    ? '[name]__[local]--[hash:base64:5]' 
                    : '[hash:base64:8]'
                }
              }
            },
            'postcss-loader'
          ]
        },

        // Images
        {
          test: /\.(png|jpe?g|gif|svg)$/i,
          type: 'asset',
          parser: {
            dataUrlCondition: {
              maxSize: 8 * 1024 // 8KB
            }
          },
          generator: {
            filename: 'images/[name].[hash][ext]'
          }
        },

        // Fonts
        {
          test: /\.(woff|woff2|eot|ttf|otf)$/i,
          type: 'asset/resource',
          generator: {
            filename: 'fonts/[name].[hash][ext]'
          }
        }
      ]
    },

    plugins: [
      // HTML template
      new HtmlWebpackPlugin({
        template: './templates/canvas.html',
        filename: 'canvas.html',
        chunks: ['main', 'workflow_canvas'],
        inject: 'body',
        minify: isProduction ? {
          removeComments: true,
          collapseWhitespace: true,
          removeRedundantAttributes: true,
          useShortDoctype: true,
          removeEmptyAttributes: true,
          removeStyleLinkTypeAttributes: true,
          keepClosingSlash: true,
          minifyJS: true,
          minifyCSS: true,
          minifyURLs: true
        } : false
      }),

      // Extract CSS in production
      ...(isProduction ? [
        new MiniCssExtractPlugin({
          filename: 'css/[name].[contenthash].css',
          chunkFilename: 'css/[name].[contenthash].chunk.css'
        })
      ] : [])
    ],

    optimization: {
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          default: {
            minChunks: 2,
            priority: -20,
            reuseExistingChunk: true
          },
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: -10,
            chunks: 'all'
          },
          react: {
            test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
            name: 'react',
            chunks: 'all',
            priority: 20
          },
          mui: {
            test: /[\\/]node_modules[\\/]@mui[\\/]/,
            name: 'mui',
            chunks: 'all',
            priority: 15
          },
          dnd: {
            test: /[\\/]node_modules[\\/](react-dnd|react-dnd-html5-backend)[\\/]/,
            name: 'dnd',
            chunks: 'all',
            priority: 10
          }
        }
      },
      
      ...(isProduction && {
        minimize: true,
        sideEffects: false
      })
    },

    devtool: isDevelopment ? 'eval-source-map' : 'source-map',

    devServer: {
      contentBase: path.join(__dirname, 'dist'),
      port: 3001,
      hot: true,
      open: true,
      historyApiFallback: true,
      compress: true,
      overlay: {
        warnings: true,
        errors: true
      },
      proxy: {
        '/api': {
          target: 'http://localhost:5000',
          changeOrigin: true,
          pathRewrite: {
            '^/api': '/api/v1/workflow_orchestration'
          }
        },
        '/ws': {
          target: 'ws://localhost:5000',
          ws: true,
          changeOrigin: true
        }
      }
    },

    performance: {
      hints: isProduction ? 'warning' : false,
      maxEntrypointSize: 512000,
      maxAssetSize: 512000
    },

    stats: {
      assets: true,
      children: false,
      chunks: false,
      hash: false,
      modules: false,
      publicPath: false,
      timings: true,
      version: false,
      warnings: true,
      colors: true
    }
  };
};